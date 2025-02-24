from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize import view
import ase
import math
import abtem
import numpy as np
import tqdm
import os
import abtem

#Functions

def get_structure_from_MPR(matarial_ids, api_key = None):
    """" 
    A function that returns an ase.atoms object interpretable by abtem
    based off stuctrure data form Materials Project Database.

    Parameters
    ----------
    material_ids : str or list of str
        String of material id/ids from Materials Project Database
    api_key: str
        Specific key for accesing Materials Project Database's API. 
        For further information visit https://next-gen.materialsproject.org/api

    Returns
    -------
    atoms : ase.Atoms
        ase.Atoms object/objects of material
    
    """
    
    if api_key is None:
        print('No api-key given - Please get api-key by logging in to Materials Project Database.')
    else:
        with MPRester("api_key") as mpr:
            docs = mpr.summary.search(material_ids=matarial_ids, fields=["structure"])
            return AseAtomsAdaptor().get_atoms(docs[0].structure)
        

def center_rotation(atoms, theta):
    """" 
    Performs a anti-clockwise rotation on a structure arounds it's center in the z-axis
    The center is defined as the center of the the atoms objects cell.

    Parameters
    ----------
    atoms : ase.Atoms
        ase.Atoms object to rotate
    theta: int or float
        number of rotation angle
    """

    rot_center = [atoms.cell[0][0]*0.5, atoms.cell[1][1]*0.5, atoms.cell[2][2]*0.5]#defining center of rotation of structure
    atoms.rotate(theta, 'z', center = rot_center) #rotation of second layer

def sampling_parameters(ase_structure, energy, resolution = 1):
    """
    Calculates TEM parameters based off the structure object, the resolution
    and the energy of the electron wave.
     Returns a dict with sampling paramateres. Units are in Å/Å^-1 and mrad
    
    Parameters
    ----------
    ase_structure : se.Atoms
        ase.Atoms object/objects of material
    energy: float or int
        Energy of electron wave given in eV. Standard values between 80,000 to 300,000 eV
    resolution: float or int
        Desired resolution of TEM image given in Å.

    Returns
    -------
    sampling metadata : dict 
        A dictionary with sampling parameters: 
          keys()
            energy: float
              Energy of electron wave given in eV.
            resolution: float
              Resolution of TEM image given in Å.
            extent: float
              extent of TEM given in Å - same in both x and y
            pixel size: float
              size of each pixel in Å - smaller than resolution due to Nyquist sampling limitaitons
            gpts: int
              number of pixels N in the NxN TEM image calculation
              always in factors of 2
            Wavelenght: float
              Wavelengt of the electron wave in Å
            Reciprocal pixel size: float
              size of each pixel in Å^-1
            k_max nyquist: float
              Limit of angular resolution due to nyquist given in Å^-1
            k_max antialiasing: float
              angular limit to avoid antialiasing in Å^-1
            Angular limitied resolution: float
              Angular limited resulotion based off k_max antialiasing given in mrad
    """
    
    f_nyquist = 0.5 # approximately double sampling rate of finest features of experiment
    Picture_size = np.max(ase_structure.cell[0:2])  #Å - taking the largest of coordinate in of the cell in the x-y plane
    delta_x = f_nyquist * resolution #pixel size length in Å
    N = Picture_size/delta_x  # number of pixels
    N = 2**math.ceil(math.log2(N)) #rounding up to nearest higher pixel count for FFT (as a factor of 2)
    delta_x = Picture_size/N # recalculating pixel size with respect to new gpts


    plane_wave = abtem.PlaneWave(gpts = N, extent=Picture_size,energy=energy)
    wavelength = plane_wave.wavelength #defining the wavelength of the source electrons given in Å

    reciprocal_P = round(1/(N*delta_x),6)
    k_max = round(f_nyquist/delta_x,6) #å^-1
    k_max_antialiasing = round(2/3 * k_max,6) 
    alpha = k_max_antialiasing * wavelength *10**3


    return {"energy":energy, "resolution":resolution, "extent":Picture_size,
            "Pixel size": delta_x, "gpts": N, "Wavelength" : wavelength,
            "Reciprocal pixel size": reciprocal_P, "k_max nyquist": k_max,
          "k_max antialiasing": k_max_antialiasing, "Angular limitied resolution": alpha}

def potential_build(ase_structure, energy, 
                    slice_thickness = 1, parametrization="lobato", 
                    projection="finite", resolution=1):
    """Builds structure potential of either Atoms or Frozen Phonons obeject. Units given in Å and eV
    
    
    Parameters
    ----------
    ase_structure : se.Atoms
        ase.Atoms object/objects of material
    energy: float or int
        Energy of electron wave given in eV. Standard values between 80,000 to 300,000 eV
    resolution: float or int
        Desired resolution of TEM image given in Å.
    parametrization: str
    projection: str
        either fintite or infinte

    Returns
    -------
    potential : abtem.Potential object
    
    """
    

    #retrives sampling parameters
    Param = sampling_parameters(ase_structure = ase_structure, energy = energy, resolution = resolution)
    
    return abtem.Potential(ase_structure, gpts=Param['gpts'], parametrization=parametrization, 
                           slice_thickness=slice_thickness, projection=projection)

def TEM_exit_wave(potential, input_wave, compute = True):
    """Returns an exit wave for an HRTEM simulation
    given input wave and potential object by using multislice approximation.
    
    Parameters
    ----------
    potential : abtem.Potential object
        potential object with slice information
    input_wave: abtem.Waves
        abtem plane wave with given parameters

    Returns
    -------
    exit_wave: abtem.Waves
        exit electron wave
    
    """
    exit_wave = input_wave.multislice(potential)
    if compute:
        exit_wave.compute()
    return exit_wave

def combine_layers(layer_1, layer_2, z_height):
    combined_layer  = layer_1 + layer_2
    combined_layer.cell[2][2] = z_height
    return combined_layer


def orthoganol_cell_cubic(ase_surface, extent, rounding_error_limit = 0.0001):
    """
    Orthogonalizes ase.surface object given a desired extent of image
    
    
    Parameters
    ----------
    ase_surface : ase.surface()
        surface from a given lattice and Miller indices.
    extent: float or int
        Size N of desired N x N in XY plane of TEM image in Å
    rounding_error_limit: float
        Error limit of where atoms are included in the lattice supercell

    Returns
    -------
    supercell : ase.surface()
        The orthogonalized supercell of the input surface
    """

    #get extent for making supercell a bit larger - gets cut down afterwards
    repetitions = repetitinos_from_min_extent(ase_surface, extent*1.2)
    supercell  = ase_surface * repetitions
    #translate cell to have bottom corner of cubic section at (0,0,0)
    if supercell.cell[1][0]>0:
        supercell.translate([-supercell.cell[1][0],0,0])
  
    #redraw the supercell matrix as square in XY
    supercell.cell[0] = [extent,0,0]
    supercell.cell[1] = [0,extent,0]

    #remove all points that fall outside the supercell
    index = 0
    while index < len(supercell.positions):
        if (min(supercell.positions[index][0:2])<= - rounding_error_limit
        or max(supercell.positions[index][0:2])> extent+rounding_error_limit):
            supercell.pop(index)
        else:
            index+=1
    return supercell


def repetitinos_from_min_extent(ase_cell, extent):
    """
    Calculates the minimum reptition of base unitcell to fill desired extent:
    
    
    Parameters
    ----------
    ase_cell : ase.Atoms or ase.surface()
        ase.Atoms object of a single unitcell or base cut of ase.surface()
    extent: float or int
        Size N of desired N x N in XY plane of TEM image in Å

    Returns
    -------
    (X,Y) : (int,int)
        Minimum of repetitions in X and Y of unitcell to achive given extent 
    """


    #get the cell vectors in the XY plane
    X = ase_cell.cell[0][0:2]
    Y = ase_cell.cell[1][0:2]

    
    #Calculate the fraction of the square section of the cell represented in the unit cell vectors 
    X_extent = abs(Y[0])/X[0]

    

    #calculate the minimum repetitions needed for full coverage
    X_rep = int( math.ceil( extent* (1/X[0]) * (1+X_extent) ) )
    Y_rep = int( math.ceil( extent / Y[1] ) ) 

    return (X_rep,Y_rep,1)

def generate_ctf(Cs = -20e-6*1e10, energy=300e3, defocus ="scherzer" ):
   """generate Contrast Transfer function to be applied on exit wave
   
   Parameters
   ----------
   Cs : float
      the sperical abberation given in Å
   energy: float
      energy of the electron wave in eV
   defocus: str or float
      The defocus setting - either automatic as set to "scherzer" or a value in Å

   Returns
   -------
   ctf : abtem.CTF
      contrast transfer function object for the given parameters 
   """
   
   
   ctf = abtem.CTF(Cs=Cs, energy=energy, defocus=defocus)

   print(f"defocus = {ctf.defocus:.2f} Å")
   aberration_coefficients = {"C10": -ctf.defocus, "C30": Cs}

   return abtem.CTF(aberration_coefficients=aberration_coefficients, energy=ctf.energy) 
   

def gen_stack(ase_cell, Theta, extent, interlayer_dist=3, xy_padding = 2,
            X_translation = [0], Y_translation = [0], sliceThickness = 0.5,
            resolution = 0.5, save_potentials = False, save_structures = False):
    """Function for getting data from many layers"""


    #making the extent of 
    non_aliasing_extent = np.ceil(extent*np.sqrt(2))
    
    #create layer 1 from repetitions and 
    layer_1 =  orthoganol_cell_cubic(ase_cell,extent=extent) 
    z_translation = [0,0,layer_1.cell[2][2]]

    #Define padding and distance between layers
    z_height = 2*interlayer_dist+ase_cell.cell[2][2]
    z_padding = interlayer_dist/2 

    #applying vacuum
    layer_1.center(vacuum = xy_padding, axis=(0,1))
    layer_1.center(vacuum = z_padding, axis=2)

    #creating layer 2
    layer_2 = layer_1.copy()
    z_translation = [0,0,layer_1.cell[2][2]] #translation of layer2
    layer_2.translate(z_translation)


    #defining the sampling parameters:
    sampling_pam = sampling_parameters(layer_1, energy = 300e3, resolution = resolution)
    #Make the ingoing plane wave:
    plane_wave = abtem.PlaneWave(energy =sampling_pam['energy'],
                                gpts = sampling_pam['gpts'],
                                extent = sampling_pam['Picture size'])




    A  = len(Theta)    
    B = len(X_translation)
    C = len(Y_translation)
    D = int(sampling_pam['gpts'])



    #Generate labels for metadata:
    labels = ["rotation offset = ", "x-axis offset = ", "y-axis offset = "]

    Y_meta_list = list(map(str, Y_translation))
    for i , lab in enumerate(Y_meta_list):
        Y_meta_list[i] = labels[2] + lab + "Å"

    X_meta_list = list(map(str, X_translation))
    for i , lab in enumerate(X_meta_list):
        X_meta_list[i] = labels[1] + lab + "Å"

    Theta_meta_list = list(map(str, Theta))
    for i , lab in enumerate(Theta_meta_list):
        Theta_meta_list[i] = labels[0] + lab + "°"

    #Data saving lists
    ase_structures = []
    potentials = []    
    
    T_s = [] #list for abtem stack creation

    for k, phi in enumerate(tqdm(Theta, desc ='Angle number:', leave=False) ):
        
        X_s = [] #list for abtem stack creation
    
        for j, x in enumerate(tqdm(X_translation, desc ='X translation:', leave=False)):
            
            Y_s = [] #list for abtem stack creation

            for l, y in enumerate(tqdm(Y_translation, desc ='Y translation:', leave=False)):
                
                layer_2 = ase_cell.copy() #make new instance of layer 2
                center_rotation(layer_2,phi) #rotate new instance
                layer_3 = orthoganol_cell_cubic(layer_2)

                combined_layers = combine_layers(layer_1,layer_3, z_height)#combine new instance with fixed layer
                if save_structures:
                    ase_structures.append(combined_layers)

                #build potential of new instance and add to list
                potential = potential_build(combined_layers, 
                                            energy=sampling_pam['energy'], 
                                            slice_thickness = sliceThickness,
                                            projection="finite", 
                                            resolution=sampling_pam['resolution'],
                                            )
                if save_potentials:
                    potentials.append(potential)

                #build exit wave and add to list
                exit_wave = TEM_exit_wave(potential, plane_wave)


                Y_s.append(exit_wave)
                
            
            X_s.append(abtem.stack(Y_s,Y_meta_list))
            
        T_s.append(abtem.stack(X_s,X_meta_list))

    exit_wave_stack = abtem.stack(T_s, Theta_meta_list)
                       
    metadata = {'Rotation' : Theta, 'x_translation' : X_translation, 'y_translation' : Y_translation, 
                'ASE structure' : ase_structures, 'Potential' : potentials}
        


    return exit_wave_stack, metadata

def save_outcome (abtem_stack:abtem.waves.Waves, filename:str):
    "Function that saves simulation outcomes"
    os.makedirs('data', exist_ok=True)
    path = os.getcwd()
    path_to_data = os.path.join(path, 'data')
    abtem_stack.to_zarr(os.path.join(path_to_data, filename))
    print(f'File has been saved to disk at directory:\n {os.path.join(path_to_data, filename)}')

def load_outcome_file(file_directory):
    "loads outcome file"
    return abtem.array.from_zarr(file_directory)
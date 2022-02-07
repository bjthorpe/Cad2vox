""" Package to provide user interface to the CudaVox package."""
import csv
from os.path import exists
import errno
import os
import numpy as np
import tifffile as tf
import meshio
from CudaVox import run
from .utill import check_greyscale,find_the_key,check_voxinfo

def voxelise(input_file,output_file,greyscale_file=None,gridsize=0,unit_length=-1.0,use_tetra=True,
             cpu=False,solid=False):
    """

    Wrapper Function to setup the CudaVox python bindings for the C++ code and provide the main user
    interface.

    Parameters:
    input_file (string): Hopefully self explanitory, Our recomended (i.e. tested) format is Salome
    med. However, theortically any of the aprrox. 30 file formats suported by meshio will
    work (see https://github.com/nschloe/meshio for the full list).

    output_file (string): Filename for output as 8 bit greyscale tiff stack.

    greyscale_file (string/None): csv file for defining custom Greyscale values. If not given the
    code evenly distributes greyscale values from 0 to 255 across all materials defined in the
    input file. It also auto-generates a file 'greyscale.csv' with the correct formatting which
    you can then tweak to your liking.

    gridsize (+ve int): Number of voxels in each axis. That is you get a grid of grisize^3 voxels
    and the resulting output will be a tiff stack of gridsize by gridside images.
    Note: if you set this to any postive interger except 0 it will calculate unit length for you
    based on the max and min of the mesh so in that case you don't set unit_length. i.e. leave
    unit_length at it's default value. (see unit_length for details).

    unit_length (+ve non-zero float): size of each voxel in mesh co-ordinate space. You can define
    this instead of Gridsize to caculate the number of voxels in each dimension, again based on max
    and min of the mesh grid. Again if using Gridsize leave this a default value (i.e. -1.0).

    use_tetra (bool): flag to specifically use Tetrahedrons instead of Triangles. This only applies
    in the event that you have multiple element types defined in the same file. Normally the code
    defaults to triangles however this flag overides that.

    cpu (bool): Flag to ignore any CUDA capible GPUS and instead use the OpenMp implementation. 
    By default the code will first check for GPUS and only use OpenMP as a fallback. This flag 
    overrides that and forces the use of OpenMP.

    Solid (bool): This Flag can be set if you want to auto-fill the interior when using a Surface
    Mesh (only applies to Triangles). If you intend to use this functionality there are three
    Caveats to briefly note here:

    1) This flag will be ignored if you only supply Tetrahedron data or set use_tetra since in
    both cases that is by definition not a surface mesh.

    2) The algorithm currently used is considerably slower and not robust (can lead to artifacts and
    holes in complex meshes).

    3) Setting this flag turns off greyscale values (background becomes 0 and the mesh becomes 255).
    This is because we dont have any data as to what materials are inside the mesh so this seems a
    sensible default.

    The only reason 2 and 3 exist is because this functionaly is not activley being used by our
    team so there has been no pressing need to fix them. However, if any of these become an
    issue either message b.j.thorpe@swansea.ac.uk or raise an issue on git repo as they can easily
    be fixed and incorporated into a future release.
    """

    # read in data from file
    mesh = meshio.read(input_file)
    # read in data from file
    mesh = meshio.read(input_file)

    #extract np arrays of mesh data from meshio
    points = mesh.points
    triangles = mesh.get_cells_type('triangle')
    tetra = mesh.get_cells_type('tetra')

    # extract dict of material names and integer tags
    all_mat_tags=mesh.cell_tags

# pull the dictionary containing material id's for the elemnt type (either triangles or tets)
# and the np array of ints that label the material in each element.
    if use_tetra:
        mat_ids = mesh.get_cell_data('cell_tags','tetra')
        mat_tag_dict = find_the_key(all_mat_tags, np.unique(mat_ids))
    else:
        mat_ids = mesh.get_cell_data('cell_tags','triangle')
        mat_tag_dict = find_the_key(all_mat_tags, np.unique(mat_ids))

    if greyscale_file is None:
        print("No Greyscale values given so they will be auto generated")
        greyscale_array = generate_greyscale(mat_tag_dict,mat_ids)
    else:
        greyscale_array = read_greyscale_file(greyscale_file,mat_ids)
    #define boundray box for mesh
    mesh_min_corner = np.array([np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])])
    mesh_max_corner = np.array([np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])])
    #check the values that have been defined by the user
    gridsize = check_voxinfo(unit_length,gridsize,mesh_min_corner,mesh_max_corner)

    #call c++ library to perform the voxelisation
    print(np.shape(greyscale_array))
    vox =(run(Triangles=triangles,Tetra=tetra,Greyscale=greyscale_array, Points=points,
       Bbox_min=mesh_min_corner,Bbox_max=mesh_max_corner,solid=solid,
        gridsize=gridsize,use_tetra=use_tetra,forceCPU=cpu)).astype('uint8')
    # write resultant 3D NP array as tiff stack
    tf.imwrite(output_file,vox,photometric='minisblack')


def generate_greyscale(mat_tags,mat_ids):
    """ Function to generate Greyscale values if none are defined"""
    # create list of tags and greyscale values for each material used
    mat_index = list(mat_tags.keys())
    mat_names = list(mat_tags.values())
    num_mats = len(mat_names)
    greyscale_values = np.linspace(255/num_mats,255,endpoint=True,num= num_mats).astype(int)
    print("writing greyscale values to greyscale.csv")
    with open('greyscale.csv', 'w',encoding='UTF-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Material Name","index","Greyscale Value"])
        for i,names in enumerate(mat_names):
            row = [" ".join(names),mat_index[i],greyscale_values[i]]
            writer.writerow(row)

    #replace the material tag in the array its the integer greyscale value
    for i,tag in enumerate(mat_index):
        mat_ids[mat_ids==tag] = greyscale_values[i]

    return mat_ids

def read_greyscale_file(greyscale_file,mat_ids):
    """ Function to Read Greyscale values from file if a file is defined by the user."""
    if not exists(greyscale_file):

        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), greyscale_file)

    with open(greyscale_file, 'r',encoding='UTF-8') as file:
        print("reading greyscale values from "+ greyscale_file)
        csv_reader = csv.DictReader(file)
        greyscale_values = []
        mat_index = []
        for i,row in enumerate(csv_reader):
        #checking the data that is being read in
            check_greyscale(row["Greyscale Value"])
            mat_index.append(int(row["index"]))
            greyscale_values.append(int(row["Greyscale Value"]))

    print(mat_index)
    #replace the material tag in the array with its the integer greyscale value
    for i,tag in enumerate(mat_index):
        mat_ids[mat_ids==tag] = greyscale_values[i]

    return mat_ids

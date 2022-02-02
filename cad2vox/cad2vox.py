from .utill import *
from cad2vox._CudaVox import run
import meshio
import numpy as np
import tifffile as tf
import csv
np.set_printoptions(threshold=np.inf)

def Voxelise(input_file,output_file,greyscale_file=None,solid=False,gridsize=0,unit_length=-1.0,use_tetra=True):
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
    if(use_tetra):
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
    
    Vox =(run(Triangles=triangles,Tetra=tetra,Tags=greyscale_array, Points=points,
       Bbox_min=mesh_min_corner,Bbox_max=mesh_max_corner,solid=solid,
        gridsize=gridsize,use_tetra=use_tetra)).astype('uint8')
    # write resultant 3D NP array as tiff stack
    tf.imwrite(output_file,Vox,photometric='minisblack')

def generate_greyscale(mat_tags,mat_ids):
    # create list of tags and greyscale values for each material used
    mat_index = list(mat_tags.keys())
    mat_names = list(mat_tags.values())
    num_mats = len(mat_names)
    greyscale_values = np.linspace(255/num_mats,255,endpoint=True,num= num_mats).astype(int)
    print("writing greyscale values to greyscale.csv")
    with open('greyscale.csv', 'w') as csvfile:
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
    from os.path import exists
    if not(exists(greyscale_file)):
        import errno
        import os
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), greyscale_file)

    with open(greyscale_file, 'r') as file:
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

#Voxelise("AMAZE_Sample.med","AMAZE_Sample.tif")
#Voxelise("NoVoid.med","NoVoid.tif")
#Voxelise("Sphere.med","Sphere.tif")

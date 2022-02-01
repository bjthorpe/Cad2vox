from _CudaVox import *
import meshio
import numpy as np
import tifffile as tf
import csv
#np.set_printoptions(threshold=np.inf)
def find_the_key(dictionary, target_keys):
    return {key: dictionary[key] for key in target_keys}

def check_greyscale(greyscale_value):
    #check if greyscale can be converted to an int
    if (not greyscale_value.isdigit()):
        print(greyscale_value)
        raise TypeError("Invalid Greyscale value, must be an Integer value")
    if ((int(greyscale_value) < 0) or (int(greyscale_value) >255)):
        raise TypeError("Invalid Greyscale value. Must be between 0 and 255")

def Voxelise(input_file,output_file,greyscale_file=None,solid=False,gridsize=0,unit_length=-1.0,use_tetra=True):
    # read in data from file
    mesh = meshio.read(input_file)

    #extract np arrays of mesh data from meshio
    points = mesh.points
    triangles = mesh.get_cells_type('triangle')
    tetra = mesh.get_cells_type('tetra')
    
    # extract dict of material names and integer tags
    all_mat_tags=mesh.cell_tags
    
    #np array of ints that label the material in each tetrahedon
    tetra_tags = mesh.get_cell_data('cell_tags','tetra')

    #pull the dictionary 
    mat_tags = find_the_key(all_mat_tags, np.unique(tetra_tags))

    if greyscale_file is None:
        print("No Greyscale values given so they will be auto generated")
        tetra_tags = generate_greyscale(mat_tags,tetra_tags,cell_type="volume")
    else:
        from os.path import exists
        if not(exists(greyscale_file)):
            import errno
            import os
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), greyscale_file)

        with open(greyscale_file, 'r') as file:
            print("reading greyscale values from "+ greyscale_file)
            csv_reader = csv.DictReader(file)
            #num_mats =sum(1 for x in csv_reader) #get total number of materials defined in csv_reader
            greyscale_values = []
            mat_index = []
            for i,row in enumerate(csv_reader):
            #checking the data that is being read in
                check_greyscale(row["Greyscale Value"])
                mat_index.append(int(row["index"]))
                greyscale_values.append(int(row["Greyscale Value"]))

        #replace the material tag in the array its the integer greyscale value
        for i,tag in enumerate(mat_index):
            tetra_tags[tetra_tags==tag] = greyscale_values[i]
    
    #define boundray box for mesh
    mesh_min_corner = np.array([np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])])
    mesh_max_corner = np.array([np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])])
    
    #call c++ library to perform the voxelisation
    Vox =(run(Triangles=triangles,Tetra=tetra,Tags=tetra_tags, Points=points, Bbox_min=mesh_min_corner,Bbox_max=mesh_max_corner,solid=solid,gridsize=gridsize,unit_length=unit_length,use_tetra=use_tetra)).astype('uint8')
    # write resultant 3D NP array as tiff stack
    tf.imwrite(output_file,Vox,photometric='minisblack')

def generate_greyscale(mat_tags,tetra_tags,cell_type):
    # create list of tags and greyscale values for each material used
    mat_index = list(mat_tags.keys())
    mat_names = list(mat_tags.values())
    num_mats = len(mat_names)
    greyscale_values = np.linspace(255/num_mats,255,endpoint=True,num= num_mats).astype(int)
    print("writing greyscale values to greyscale.csv")
    with open('greyscale.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Material Name","cell type","index","Greyscale Value"])
        for i,names in enumerate(mat_names):
            row = [" ".join(names),cell_type,mat_index[i],greyscale_values[i]]
            writer.writerow(row)

    #replace the material tag in the array its the integer greyscale value
    for i,tag in enumerate(mat_index):
        tetra_tags[tetra_tags==tag] = greyscale_values[i]

    return tetra_tags

Voxelise("AMAZE_Sample.med","AMAZE_Sample.tif",gridsize=200)
#Voxelise("NoVoid.med","NoVoid.tif")
#Voxelise("Sphere.med","Sphere.tif")

"""A bunch of useful utility functions"""
import numpy as np
#check if greyscale can be converted into a vaild 8-bit int
def check_greyscale(greyscale_value):
    """ Function to check the user defined Greyscale values are valid 8-bit integers"""
    try:
        int(greyscale_value)
    except ValueError:
        print(greyscale_value)
        raise ValueError("Invalid Greyscale value, must be an Integer value, "
        "or castable to and Integer value")

    if ((int(greyscale_value) < 0) or (int(greyscale_value) >255)):
        raise ValueError("Invalid Greyscale value. Must be between 0 and 255")

def find_the_key(dictionary, target_keys):
    """Function to pull out the keys of a dictionary as a list."""
    return {key: dictionary[key] for key in target_keys}

def check_gridsize(gridsize):
    """check that gridsize is a list of 3 non-zero positive integers"""
    if not isinstance(gridsize, list):
        raise TypeError("Invalid Gridsize. Must be a list.")
    if (len(gridsize) != 3):        
            raise TypeError("Invalid Gridsize. Must be a list of three integer values.")

    for i in gridsize:
        if not isinstance(i, int):
            raise TypeError(f"Invalid Gridsize {i}. Must be an integer value.")
        if i < 0:
            raise TypeError(f"Invalid Gridsize {i}. Must be an integer value that is"
                        "greater than 0.")

def check_unit_length(unit_length):
    """check that unit_length is a list of three non-zero positive floats."""
    if not isinstance(unit_length, list):
        raise TypeError("Invalid unit_length. Must be a list.")
    if (len(unit_length) != 3):        
            raise TypeError("Invalid unit_length. Must be a list of three floats.")

    for i in unit_length:
        if not isinstance(i, float):
            raise TypeError(f"Invalid unit length {i} Must be an floating point value.")
        if i < 0:
            raise TypeError(f"Invalid unit length {i}. Must be an floating point value"
                            " that is greater than 0.")

def check_voxinfo(unit_length=[0.0,0.0,0.0],gridsize=[0,0,0],Bbox_centre="mesh",
                  mesh_min=None,mesh_max=None):
    """

    check to see that both unit_length and gridsize are valid and at least one is defined.
    After which it calculates the size of the image boundary box.

     

    Note: the calculation of unit_length is in reality handled deep in the C++
    code. This is because the code incorporates a small displacement to the grid
    "epsilon" to try and avoid vertices falling directly on the grid
    (see "util.h" line 98). Thus this function is a bit of a hack in that it
    although we calculate the unit_length we never actually use it.

    It just hands gridsize off to the c++ code to calculate the "actual" unit_length.

    Note this function converts gridsize from a list to an np array in the final step.

    """
    import numpy as np
    #  check gridsize and unit_length are valid.
    check_gridsize(gridsize)
    check_unit_length(unit_length)

    if((gridsize==[0,0,0]) and (0.0 not in unit_length)):
    # unit_length has been defined by user so check it is valid and
    # then calculate gridsize.
        for i,_ in enumerate(gridsize):
            gridsize[i] = int((mesh_max[i] - mesh_min[i])/ unit_length[i])
            Bbox_min = mesh_min
            Bbox_max= mesh_max
    # GridSize has been defined by user so check it is valid and
    # then calculate unit_length.
    elif((unit_length == [0.0,0.0,0.0]) and (0 not in gridsize)):
        for i,_ in enumerate(gridsize):
            unit_length[i] = float((mesh_max[i] - mesh_min[i])/ gridsize[i])
        Bbox_min = mesh_min
        Bbox_max= mesh_max

    elif((gridsize==[0,0,0]) and (unit_length==[0.0,0.0,0.0])):
        #Neither has been defined
        raise TypeError("You must define at least one of either Gridsize or unit_length")

    else:
        #Both have been defined by user in which case we calculate the image boundary's
        if Bbox_Centre == 'mesh':
            Bbox_Centre = (mesh_max-mesh_min)/2
        Bbox_min = Bbox_centre - ((np.array(gridsize)/2)*unit_length)
        Bbox_max = Bbox_centre + ((np.array(gridsize)/2)*unit_length)
    
    gridsize= np.array(gridsize)
    Vox_info = {'gridsize':gridsize,
                "unit_length":unit_length,
                "Bbox_centre":Bbox_centre,
                "Bbox_min":Bbox_min,
                "Bbox_max":Bbox_max}
    return Vox_info
    

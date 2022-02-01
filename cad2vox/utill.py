# a bunch of useful utility functions
import numpy as np
#check if greyscale can be converted into a vaild 8-bit int
def check_greyscale(greyscale_value):
    if (not greyscale_value.isdigit()):
        print(greyscale_value)
        raise TypeError("Invalid Greyscale value, must be an Integer value")
    if ((int(greyscale_value) < 0) or (int(greyscale_value) >255)):
        raise TypeError("Invalid Greyscale value. Must be between 0 and 255")

def find_the_key(dictionary, target_keys):
    return {key: dictionary[key] for key in target_keys}

def check_gridsize(gridsize):
    if (not (isinstance(gridsize, int))):
        raise TypeError("Invalid Gridsize. Must be an integer value.")
    if(gridsize < 0):
        raise TypeError("Invalid Gridsize. Must be an integer value that is greater than 0.")

def check_unit_length(unit_length):
    if (not (isinstance(unit_length, float))):
        raise TypeError("Invalid unit length. Must be an floating point value.")
    if(unit_length <= 0):
        raise TypeError("Invalid unit length. Must be an floating point value that is greater than 0.")

def check_voxinfo(unit_length,gridsize,gridmin,gridmax):
    if((gridsize==0) and (unit_length!=-1.0)):
    #unit_length has been defined by user so check it is valid and then calulate gridsize
        check_unit_length(unit_length)
        gridsize = int((np.max(gridmax) - np.min(gridmin))/ unit_length)
        print("gridsize =", gridsize)

    elif((gridsize!=0) and (unit_length==-1.0)):
    #gridsize has been defined by user so check it is valid
        check_gridsize(gridsize)
    
    elif((gridsize==0) and (unit_length==-1.0)):
    #Neither has been defined
        raise TypeError("You must define one (and only one) of either Gridsize or unit_length")

    else:
    #Both have been defined by user in which case throw an error as we need at least one of them to calculate the other.
        raise TypeError("Both Gridsize and unit length appear to have been defined by the user. Please only define one.")

    return gridsize

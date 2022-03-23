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
    """check that gridsize is a non-zero positive integer"""
    if not isinstance(gridsize, int):
        raise TypeError("Invalid Gridsize. Must be an integer value.")
    if gridsize < 0:
        raise TypeError("Invalid Gridsize. Must be an integer value that is"
                        "greater than 0.")

def check_unit_length(unit_length):
    """check that unit_length is a non-zero positive float."""
    if not isinstance(unit_length, float):
        raise TypeError("Invalid unit length. Must be an floating point value.")
    if unit_length < 0:
        raise TypeError("Invalid unit length. Must be an floating point value"
                        " that is greater than 0.")

def check_voxinfo(unit_length,gridsize,gridmin,gridmax):
    """

    check to see that both unit_length and gridsize are valid and have not both
    been defined at the same time.

    Note: the calculation of unit_length is in reality handled deep in the C++
    code. This is because the code incorporates a small displacement to the grid
    "epsilon" to try and avoid vertices falling directly on the grid
    (see "util.h" line 98). Thus this function is a bit of a hack in that it
    always returns just the gridsize. Either the user supplied value
    (assuming it is valid) or the calculated value from the unit length.
    It then hands that off to the c++ code to calculate the
    "actual" unit_length.

    """
    if((gridsize==0) and (unit_length!=0.0)):
    # unit_length has been defined by user so check it is valid and
    # then calulate gridsize.
        check_unit_length(unit_length)
        gridsize = int((np.max(gridmax) - np.min(gridmin))/ unit_length)
        print("gridsize =", gridsize)

    elif((gridsize!=0) and (unit_length==0.0)):
    # gridsize has been defined by user so check it is valid.
    # Note: The calculation of unit_length is handled in c++.
        check_gridsize(gridsize)

    elif((gridsize==0) and (unit_length==0.0)):
    #Neither has been defined
        raise TypeError("You must define one (and only one) of either Gridsize"
                        "or unit_length")

    else:
    #Both have been defined by user in which case throw an error as we need
    # at least one of them to calculate the other.
        raise TypeError("Both Gridsize and unit length appear to have been"
                        " defined by the user. Please only define one.")

    return gridsize

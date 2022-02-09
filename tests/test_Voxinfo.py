""" Various sanity tests to check that filenames are handled corectly."""
import pytest
import cad2vox
import os

# use the test case folder as their working directory
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

@pytest.fixture()
def cleanup():
    print("Starting Pytest")
    yield
    print("performing cleanup of output")
    if os.path.exists("greyscale.csv"):
        os.remove("greyscale.csv")

############ tests for unit length

@pytest.mark.parametrize("input_length",[5,"max","0.012",-0.00001])
def test_invalid_unit_length(input_length):
    # Test -ve or non-float values gives error
    with pytest.raises(TypeError):
        cad2vox.voxelise("inputs/AMAZE_Sample.med","outputs/AMAZE_Sample.tiff",unit_length=input_length)
    assert TypeError

@pytest.mark.parametrize("gridsize",[-5,"max","0.012",0.00001])
def test_invalid_gridsize(gridsize):
    # Test negative or non int value for gridsize gives error
    with pytest.raises(TypeError):
        cad2vox.voxelise("inputs/AMAZE_Sample.med","outputs/AMAZE_Sample.tiff",gridsize=gridsize)
    assert TypeError

def test_set_both():
    # Test negative or non int value for gridsize gives error
    with pytest.raises(TypeError):
        cad2vox.voxelise("inputs/AMAZE_Sample.med","outputs/AMAZE_Sample.tiff",gridsize=100,unit_length=0.0001)
    assert TypeError
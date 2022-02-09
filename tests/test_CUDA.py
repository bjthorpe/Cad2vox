""" Test to see if CUDA is installed corectly and a GPU* can be found."""
import pytest
from CudaVox import Check_CUDA

@pytest.mark.xfail
def test_CUDA():
    # Call function to check for CUDA capable GPU
    result = Check_CUDA()
    assert result
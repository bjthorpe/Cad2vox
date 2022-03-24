""" Test to see if CUDA is installed corectly and a GPU* can be found."""
import pytest


@pytest.mark.CUDA
def test_CUDA():
    from CudaVox import Check_CUDA
    # Call function to check for CUDA capable GPU
    result = Check_CUDA()
    assert result
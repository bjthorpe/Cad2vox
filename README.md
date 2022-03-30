## Details
Cad2Vox is a Python package to efficiently perform mesh voxelisation on GPU (using CUDA) or CPU (using OpenMP) for surface and volume cad meshes based on triangles and tetrahedrons respectively.

The code itself consists of two Python packages, cad2vox and Cudavox. CudaVox is a python package that provides bindings to c++ code using pybind11 and xtensor python (using CUDA to perform calculations on a GPU and OpenMP to perfrom them in parallel on a multi-core CPU). Cad2vox meanwhile is a pure python package that reads in and wrangles the mesh/greyscale data using meshio and acts as a user interface to Cudavox.

This project is a fork of cuda_voxelizer (https://github.com/Forceflow/cuda_voxelizer) the original plan was to simply add bindings to allow us to call it from python. However as my research project progressed the code has since ballooned into its own thing (adding support for volume meshes and using meshio and xtensor instead of trimesh). Thus it feels more appropriate to release it as a standalone project.

For Surface meshes (based on triangles) CudaVox implements an optimised version of the method described in M. Schwarz and HP Seidel's 2010 paper [*Fast Parallel Surface and Solid Voxelization on GPU's*](http://research.michael-schwarz.com/publ/2010/vox/).

For volume meshes (based on Tetrahedrons) it uses a simple algorithm to check if a point P (taken as the centre of the voxel) is inside a tetrahedron defined by 4 vertices (A,B,C,D). This is achieved by calculating  the normal of the four triangles that make up the surface of the tetrahedron. Since these vectors will all point away from the centre of the tetrahedron we can simply check to see if the point P is on the opposite side of the plane for each of the four triangles. if this is true for all 4 planes then the point must be inside the tetrahedron (see https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not/51733522#51733522 for examples of this algorithm implemented in python).

## Building and Installing

### Dependencies
The project has the following build dependencies:
 * [GLM](http://glm.g-truc.net/0.9.8/index.html) for vector math. Any recent version will do.
 * [OpenMP](https://www.openmp.org/)
 * [Python](https://www.python.org/) version 3.6 or higher.

It also has the following optional dependency (see building without CUDA for details):
* [Nvidia Cuda 8.0 (or higher)](https://developer.nvidia.com/cuda-toolkit) for CUDA + Thrust libraries.

 You will also need the following python packages:
 * cmake
 * numpy
 * pybind11
 * tifffile
 * xtensor
 * xtl
 * xtensor-python
 * meshio
 * pytest
 * pandas
 * pillow

We recommend using [anaconda](https://anaconda.org/) as the python dependencies are all available through the conda package manager (as well as CUDA through cudatookit). These can be installed with the following two commands.

```bash
conda install cmake numpy pybind11 tifffile pillow cudatoolkit

conda install -c conda-forge xtensor xtl meshio xtensor-python
```

If however, you wish to use pure python many of the packages are available through pip and you can obtain them using:
```bash
pip install -r requirements.txt
```
You will however, need to build xtl, xtensor and xtensor-python from source using cmake. Instructions for which can be found here:

* [Xtl](https://github.com/xtensor-stack/xtl)
* [Xtensor](https://github.com/xtensor-stack/xtensor)
* [Xtensor-python](https://github.com/xtensor-stack/xtensor-python)

Once you have the dependencies installed you can use the setup.py scripts to build and install the two packages as:

```python
python3 Cudavox/setup_cudavox.py install

python3 setup_cad2vox.py install
```
### Building without CUDA

It is possible to build cad2vox without CUDA or a GPU. To do this simply install all the requirments except CUDA (or cudatoolkit if using ananconda)
and build as above i.e.

```python
python3 Cudavox/setup_cudavox.py install

python3 setup_cad2vox.py install
```

When building CudaVox Cmake will detect if CUDA is installed and configured corectly. If Cmake finds a sutible instalation of CUDA It will then automatically include all the aditional sorce and headerfiles nessacry to perfrom caculations on ether the GPU or CPU. If CUDA is not installed it will compile CudaVox to do caculations on the CPU only, using OpenMP. This is intened to provide options to the end user as CUDA is a rather large (not to mention proprietary) depenedecy that some users may not want/need.

The OpenMP version is by comparison considerably slower, however, OpenMP is much more compatible and not tied to Nvidia hardware (pretty much any modern CPU and compiler should suport OpenMP out of the box). OpenMP is also useful for calculationbs that wont fit in VRAM. Given that the memory requirements are in our experiance the main bottleneck and the required memory also scales cubically with Gridsize. Memory can very quickly become a limiting factor.

## Automated testing

It is good practice once Cad2vox is built and installed to test the functionallity. To this end we have included an automated test suite using pytest that can be run as follows in the root directory of cad2vox:

```bash
pytest
```
This will test the code under a variety of differnt senarios and if your setup is working correctly should all pass (for those curious souls who wish to see what we are testing the test functions are stored in the tests sub directory).

Note: if you are **NOT USING CUDA** some of the tests may fail. However, you can skip any tests related to CUDA with:

```bash
pytest -k "not CUDA"
```

## Usage

The main user facing python function from Cad2Vox is voxelise.

The information about how to call/use the voxelise function can be viewed at anytime through python by calling:

```python
import cad2vox
help(cad2vox.voxelise)
```
    voxelise(input_file, output_file, greyscale_file=None, gridsize=0, unit_length=-1.0, use_tetra=False, cpu=False, solid=False, im_format=None)
    
    Wrapper Function to setup the CudaVox python bindings for the C++ code and provide the main user
    interface.

    This function will first try to perform the voxelisation using a CUDA capible GPU. If that fails
    or CUDA is unavalible it will fallback to running on CPU with the maximum number of avalible 
    threads.
    
    Parameters:
    input_file (string): Hopefully self explanitory, Our recomended (i.e. tested) format is Salome
    med. However, theortically any of the aprrox. 30 file formats suported by meshio will
    work. Provided they are using either tetrahedrons or triangles as there element type
    (see https://github.com/nschloe/meshio for the full list).
    
    output_file (string): Filename for output as 8 bit greyscale images. Note do not include the
    extension as it will automatically be appended based on the requested image format.
    The default is a virtual tiff stack other formats can be selected with the im_format option.
    
    greyscale_file (string/None): csv file for defining custom Greyscale values. If not given the
    code evenly distributes greyscale values from 0 to 255 across all materials defined in the
    input file. It also auto-generates a file 'greyscale.csv' with the correct formatting which
    you can then tweak to your liking.
    
    gridsize (+ve int): Number of voxels in each axis. That is you get a grid of grisize^3 voxels
    and the resulting output will be a series of gridsize by gridside images. Note: if you set
    this to any postive interger except 0 it will calculate unit length for you based on the max
    and min of the mesh so in that case you don't set unit_length. i.e. leave unit_length at
    it's default value. (see unit_length for details).

    unit_length (+ve non-zero float): size of each voxel in mesh co-ordinate space. You can define
    this instead of Gridsize to caculate the number of voxels in each dimension, again based on max
    and min of the mesh grid. Again if using Gridsize leave this a default value (i.e. -1.0).
    
    use_tetra (bool): flag to specifically use Tetrahedrons instead of Triangles. This only applies
    in the event that you have multiple element types defined in the same file. Normally the code
    defaults to triangles however this flag overides that.
    
    cpu (bool): Flag to ignore any CUDA capible GPUS and instead use the OpenMp implementation.
    By default the code will first check for GPUS and only use OpenMP as a fallback. This flag
    overrides that and forces the use of OpenMP. Note: if you wish to use CPU permenantly, 
    as noted in the build docs, you can safely compile CudaVox without CUDA in which case the code
    simply skips the CUDA check altogether and permenantly runs on CPU.
    
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

    im_format (string): The default output is a Tiff virtual stack written using tiffle. This option
    however, when set allows you to output each slice in z as a seperate image in any format supported
    by Pillow (see https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html for the full
    list). Simply specify the format you require as a sting e.g. "png" Note: this has only been fully tested 
    with png and jpeg so your millage may vary.
    
## Citation
If you use Cad2Vox in your published paper or other software, please reference it, for example as follows:
<pre>
@Misc{CAD2VOX,
author = "Dr Benjamin Thorpe",
title = "Cad2Vox",
howpublished = "\url{https://github.com/bjthorpe/Cad2vox}",
year = "2022"}
</pre>
If you end up using Cad2Vox in something cool I'd be interested to hear about it so feel free to drop me an e-mail: **b,.j.thorpe@swansea.ac.uk**

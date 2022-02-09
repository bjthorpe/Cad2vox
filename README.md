## Details
Cad2Vox and CudaVox are Python packages to efficiently perform mesh voxelisation on GPU (using CUDA) or CPU (using OpenMP) for surface and volume cad meshes based on triangles and tetrahedrons respectively.

The code itself consists of two Python packages. CudaVox a python package built from the c++ code using pybind11 and cad2vox which is a pure python package that reads in and wrangles the mesh/greyscale data using meshio and acts as a user interface to Cudavox.

This project is a fork of cuda_voxelizer (https://github.com/Forceflow/cuda_voxelizer) the original plan was to simply add bindings to allow us to call it from python. However as my research project progressed the code has since ballooned into its own thing (adding support for volume meshes and using meshio and xtensor instead of trimesh). Thus it feels more appropriate to release it as a standalone project.

For Surface meshes (based on triangles) CudaVox implements an optimised version of the method described in M. Schwarz and HP Seidel's 2010 paper [*Fast Parallel Surface and Solid Voxelization on GPU's*](http://research.michael-schwarz.com/publ/2010/vox/).

For volume meshes (based on Tetrahedrons) it uses a simple algorithm to check if a point P (taken as the centre of the voxel) is inside a tetrahedron defined by 4 vertices (A,B,C,D). This is achieved by calculating  the normal of the four triangles that make up the surface of the tetrahedron. Since these vectors will all point away from the centre of the tetrahedron we can simply check to see if the point P is on the opposite side of the plane for each of the four triangles. if this is true for all 4 planes then the point must be inside the tetrahedron (see https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not/51733522#51733522 for examples of this algorithm implemented in python).

## Building and Installing

### Dependencies
The project has the following build dependencies:
 * [Nvidia Cuda 8.0 (or higher)](https://developer.nvidia.com/cuda-toolkit) for CUDA + Thrust libraries (standard included)
 * [GLM](http://glm.g-truc.net/0.9.8/index.html) for vector math. Any recent version will do.
 * [OpenMP](https://www.openmp.org/)
 * [Python](https://www.python.org/) version 3.6 or higher.
 
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

We recommend using [anaconda](https://anaconda.org/) as they are all available through the conda package manager and can be installed with the following two commands.

```bash
conda install cmake numpy pybind11 tifffile

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
python3 setup_cudavox.py install

python3 setup_cad2vox.py install
```

## Usage

The main user facing function from Cad2Vox is voxelise. The following can be viewed at anytime through python by calling:

```python
import cad2vox
help(cad2vox.voxelise)
```

voxelise(input_file, output_file, greyscale_file=None, gridsize=0, unit_length=-1.0, use_tetra=True, cpu=False, solid=False)

    Parameters:
    input_file (string): Hopefully self explanatory, Our recommended (i.e. tested) format is Salome
    med. However, theoretically any of the approx. 30 file formats supported by meshio will
    work. Provided they are using either tetrahedrons or triangles as there element type
    (see https://github.com/nschloe/meshio for the full list).

    output_file (string): Filename for output as 8 bit grey-scale tiff stack.

    grey-scale_file (string/None): csv file for defining custom Grey-scale values. If not given the
    code evenly distributes grey-scale values from 0 to 255 across all materials defined in the
    input file. It also auto-generates a file 'greyscale.csv' with the correct formatting which
    you can then tweak to your liking.

    gridsize (+ve int): Number of voxels in each axis. That is you get a grid of grisize^3 voxels
    and the resulting output will be a tiff stack of gridsize by gridside images.
    Note: if you set this to any positive integer except 0 it will calculate unit length for you
    based on the max and min of the mesh so in that case you don't set unit_length. i.e. leave
    unit_length at it's default value. (see unit_length for details).

    unit_length (+ve non-zero float): size of each voxel in mesh co-ordinate space. You can define
    this instead of Gridsize to calculate the number of voxels in each dimension, again based on max
    and min of the mesh grid. Again if using Gridsize leave this a default value (i.e. -1.0).

    use_tetra (bool): flag to specifically use Tetrahedrons instead of Triangles. This only applies
    in the event that you have multiple element types defined in the same file. Normally the code
    defaults to triangles however this flag overrides that.

    cpu (bool): Flag to ignore any CUDA capable GPUS and instead use the OpenMp implementation. 
    By default the code will first check for GPUS and only use OpenMP as a fallback. This flag 
    overrides that and forces the use of OpenMP.

    Solid (bool): This Flag can be set if you want to auto-fill the interior when using a Surface
    Mesh (only applies to Triangles). If you intend to use this functionality there are three
    Caveats to briefly note here:

    1) This flag will be ignored if you only supply Tetrahedron data or set use_tetra since in
    both cases that is by definition not a surface mesh.

    2) The algorithm currently used is considerably slower and not robust (can lead to artefacts and
    holes in complex meshes).

    3) Setting this flag turns off grey-scale values (background becomes 0 and the mesh becomes 255).
    This is because we don't have any data as to what materials are inside the mesh so this seems a
    sensible default.

    The only reason 2 and 3 exist is because this functionally is not actively being used by our
    team so there has been no pressing need to fix them. However, if any of these become an
    issue either message b.j.thorpe@swansea.ac.uk or raise an issue on git repo as they can easily
    be fixed and incorporated into a future release.

 
## Citation
If you use Cad2Vox in your published paper or other software, please reference it, for example as follows:
<pre>
@Misc{CAD2VOX,
author = "Dr Benjamin Thorpe",
title = "Cad2Vox",
howpublished = "\url{https://github.com/bjthorpe/Cad2vox}",
year = "2022"}
</pre>
If you end up using Cad2Vox in something cool, drop me an e-mail: **b,.j.thorpe@swansea.ac.uk**

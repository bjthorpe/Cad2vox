import setuptools
import os
import re
import sys
import sysconfig
import platform
import subprocess
import numpy
from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

'''
Modified from https://www.benjack.io/2017/06/12/python-cpp-tests.html
'''
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='',sources=[]):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']
        cmake_args += ['-DNUMPY_DIR=' + numpy.get_include()]
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)

with open("../README.md", "r") as f:
    long_description = f.read()

setup(
    name="CudaVox",
    version="1.5.6",
    author="Ben Thorpe",
    author_email="b.j.thorpe@swansea.ac.uk",
    description="Python bindings for a C++ library to convert mesh models into Voxel images with OpenMp and CUDA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bjthorpe/Cad2vox',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux"
    ],
    include_dirs=[numpy.get_include()],
    ext_modules=[CMakeExtension('CudaVox',sources=["python_bind.cpp"])],
    python_requires='>=3.6',
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=['numpy>=1.18','meshio','tifffile', 'pillow>=8.3', 'pandas', 'cmake>=3.22.0','pybind11'],
)

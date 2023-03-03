from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="cad2vox",
    version="1.7.3",
    author="Ben Thorpe",
    author_email="b.j.thorpe@swansea.ac.uk",
    description="A python library to provide the user interface for the CudaVox library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bjthorpe/Cad2vox',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux"
    ],
    python_requires='>=3.6',
    zip_safe=False,
    install_requires=['CudaVox>=1.7','numpy>=1.19','meshio','tifffile', 'pillow>=8.3', 'pandas', 'cmake>=3.22.0','pybind11'],
    packages=['cad2vox']
)

from setuptools import setup

setup(
    name="cad2vox",
    version="1.0.0",
    author="Ben Thorpe",
    author_email="b.j.thorpe@swansea.ac.uk",
    description="A python library to provide the user interface for the CudaVox library",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux"
    ],
    python_requires='>=3.6',
    zip_safe=False,
    install_requires=['numpy>=1.18','meshio','CudaVox'],
    packages=['cad2vox']
)

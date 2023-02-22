#!/bin/bash
#####################################
# to be run inside cad2vox container.
# with source code directory mounted as /src
#####################################
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include
cd /src/Cad2vox/CudaVox
##################################
# build and fix cudavox
##################################
for PYBIN in /opt/python/*/bin; do
"${PYBIN}/pip" install numpy
"${PYBIN}/pip" wheel . -w /output
done

for whl in /output/CudaVox*.whl; do
    auditwheel repair "$whl" -w /output/fixed
done
#######################################
cd ..

# build Cad2Vox
for PYBIN in /opt/python/*/bin; do
"${PYBIN}/pip" wheel . -w /output
done

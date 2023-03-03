#!/bin/bash
#####################################
# to be run inside cad2vox container.
# with source code directory mounted as /src
#####################################
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include
cd /src/Cad2vox/
#build Cad2Vox
for PYBIN in /opt/python/*/bin; do
"${PYBIN}/pip" wheel . -w /output
done

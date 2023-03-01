#!/bin/bash
# upload cudavox wheels
for WHEEL in CudaVox*.whl; do
    twine upload "$WHEEL"
done
# upload cad2vox wheels
for WHEEL in Cad2vox*.whl; do
    twine upload "$WHEEL"
done

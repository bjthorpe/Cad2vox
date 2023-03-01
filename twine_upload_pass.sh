#!/bin/bash
echo -n "Enter Twine username: " 
read USER
echo -n "Enter Twine Password: "
stty -echo
read PASS
stty echo
# upload cudavox wheels
for WHEEL in CudaVox*.whl; do
    expect script.exp "$USER" "$PASS" "$WHEEL"
done
# upload cad2vox wheels
for WHEEL in Cad2vox*.whl; do
    expect script.exp "$USER" "$PASS" "$WHEEL"
done

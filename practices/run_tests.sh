#!/bin/bash

# tests all the practices

for dir in practices/*; do
    # go into the dir
    cd $dir
    
    # run the install script
    chmod +x install.sh
    bash install.sh

    # test the script
    python test_*.py

    # cleanup
    rm -rf venv
    cd ..
done
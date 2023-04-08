#!/bin/bash

# tests all the practices
cd practices

for dir in ./*; do
    if [ -d "$dir" ]; then
        echo "Testing $dir"
        cd $dir
    
        # run the install script
        chmod +x install.sh
        ./install.sh

        # activate the virtual environment
        source venv/bin/activate

        # test the script
        python test.py

        # deactivate the virtual environment
        deactivate

        # cleanup
        rm -rf venv
        cd ..
    fi
done
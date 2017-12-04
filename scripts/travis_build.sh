#!/usr/bin/env bash

export NUM_JOBS=1

mkdir build
cd build
cmake ..
make
bin/unit_tests
if [[ "$unamestr" == 'Darwin' ]]; then
    echo "Disabling pip test for macOS"
else
cd ..
    pip install --user -r requirements.txt
    pip install --user .
    python src/tests/python_tests/main.py
fi

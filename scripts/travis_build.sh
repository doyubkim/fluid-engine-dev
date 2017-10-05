#!/usr/bin/env bash

mkdir build
cd build
cmake ..
make
bin/unit_tests
cd ..
pip install --user -r requirements.txt
pip install --user .
python src/tests/python_tests/main.py

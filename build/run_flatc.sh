#!/bin/bash

flatc --cpp -o src/jet src/jet/basic_types.fbs
flatc --cpp -o src/jet src/jet/grid_system_data2.fbs
flatc --cpp -o src/jet src/jet/grid_system_data3.fbs

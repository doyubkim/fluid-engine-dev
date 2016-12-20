#!/bin/bash

for file in src/jet/schema/*.fbs; do
    flatc -c -o src/jet/generated "$file"
done

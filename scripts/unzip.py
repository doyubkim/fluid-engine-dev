#!/usr/bin/env python

"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import sys
import zipfile

def main():
    input_zip = sys.argv[1]
    output_dir = sys.argv[2]

    with zipfile.ZipFile(input_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

if __name__ == '__main__':
    main()

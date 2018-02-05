#!/usr/bin/env python

"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import filecmp
import inspect
import shutil
import os
import utils

dirname = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))


def main():
    include_dir = os.path.join(dirname, "../include/jet")
    filenames = utils.get_all_files(include_dir, ["*.h"])
    filenames.sort()
    header = os.path.join(dirname, "../include/jet/jet.h")
    header_tmp = header + ".tmp"
    with open(header_tmp, "w") as header_file:
        header_file.write("""// Copyright (c) 2018 Doyub Kim

// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.\n
""")
        header_file.write("#ifndef INCLUDE_JET_JET_H_\n")
        header_file.write("#define INCLUDE_JET_JET_H_\n")
        for filename in filenames:
            if not filename.endswith("-inl.h"):
                line = "#include <jet/%s>\n" % os.path.basename(filename)
                header_file.write(line)
        header_file.write("#endif  // INCLUDE_JET_JET_H_\n")
    if not filecmp.cmp(header, header_tmp):
        shutil.move(header_tmp, header)
    else:
        os.remove(header_tmp)


if __name__ == "__main__":
    main()

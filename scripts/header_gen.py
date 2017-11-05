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
import sys
import utils

dirname = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))


def main():
    module_name = "jet" if len(sys.argv) <= 1 else sys.argv[1]
    module_name_upper = module_name.upper().replace(".", "_")

    include_dir = os.path.join(dirname, "../include/" + module_name)
    filenames = utils.get_all_files(include_dir, ["*.h"])
    filenames.sort()
    header = os.path.join(dirname, "../include/%s/%s.h" %
                          (module_name, module_name))
    header_tmp = header + ".tmp"
    with open(header_tmp, "w") as header_file:
        header_file.write("""// Copyright (c) 2018 Doyub Kim

// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.\n
""")
        header_file.write("#ifndef INCLUDE_%s_%s_H_\n" %
                          (module_name_upper, module_name_upper))
        header_file.write("#define INCLUDE_%s_%s_H_\n" %
                          (module_name_upper, module_name_upper))
        for filename in filenames:
            if not filename.endswith("-inl.h") and not filename.endswith("-ext.h"):
                line = "#include <%s/%s>\n" % (module_name, os.path.basename(filename))
                header_file.write(line)
        header_file.write("#endif  // INCLUDE_%s_%s_H_\n" %
                          (module_name_upper, module_name_upper))
    if not filecmp.cmp(header, header_tmp):
        shutil.move(header_tmp, header)
    else:
        os.remove(header_tmp)


if __name__ == "__main__":
    main()

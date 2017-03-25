import utils

def main():
#     for path in utils.navigate_all_files('include', ['*.cpp', '*.h']):
#         with open(path, 'r') as file:
#             filedata = file.read()
#         new_copyright = """// Copyright (c) 2017 Doyub Kim
# //
# // I am making my contributions/submissions to this project solely in my
# // personal capacity and am not conveying any rights to any intellectual
# // property of any third parties."""
#         filedata = filedata.replace('// Copyright (c) 2017 Doyub Kim', new_copyright)
#         filedata = filedata.replace('// Copyright (c) 2016 Doyub Kim', new_copyright)
#         with open(path, 'w') as file:
#             file.write(filedata)

    for path in utils.navigate_all_files('src', ['SConscript']):
        with open(path, 'r') as file:
            filedata = file.read()
        new_copyright = """Copyright (c) 2017 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties."""
        filedata = filedata.replace('Copyright (c) 2017 Doyub Kim', new_copyright)
        filedata = filedata.replace('Copyright (c) 2016 Doyub Kim', new_copyright)
        with open(path, 'w') as file:
            file.write(filedata)

if __name__ == '__main__':
    main()

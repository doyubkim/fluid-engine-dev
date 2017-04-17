#
# Copyright (c) 2017 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# Target name
set(target pyjet)

# Sources
file(GLOB sources
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Add Pybind11 module
pybind11_add_module(${target} ${sources})

# Link libraries
target_link_libraries(${target}
        PRIVATE
        ${DEFAULT_LINKER_OPTIONS}
        jet)

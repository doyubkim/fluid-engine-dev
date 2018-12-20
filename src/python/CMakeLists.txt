#
# Copyright (c) 2018 Doyub Kim
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

# Disable deprecated-register warning for Python2.7 headers
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif ()

# Add Pybind11 module
pybind11_add_module(${target} ${sources})

# Project options
set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS})

# Includes
target_include_directories(${target}
    PRIVATE
    ${JET_INCLUDE_DIR}
    ${PYBIND11_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(${target}
        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        jet)

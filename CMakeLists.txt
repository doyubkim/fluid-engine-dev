#
# Copyright (c) 2017 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# CMake version
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Declare project
project(jet)

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Includes
include_directories(include)
include_directories(external)
include_directories(external/pybind11/include)
include_directories(external/googletest/googletest/include)
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    include_directories(src/winix)
endif()

# Compile options
include(cmake/CompileOptions.cmake)

# Build type - Release by default
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Overrides
set(CMAKE_MACOSX_RPATH ON)

# Unpack resources
set(RESOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/resources)
set(RESOURCES_OBJS
        ${RESOURCES_DIR}/bunny.obj
        ${RESOURCES_DIR}/cube.obj
        ${RESOURCES_DIR}/cup.obj
        ${RESOURCES_DIR}/dragon.obj)
set(unzip_py ${CMAKE_CURRENT_SOURCE_DIR}/scripts/unzip.py)
add_custom_command(
        COMMAND python ${unzip_py} ${RESOURCES_DIR}/resources.zip ${RESOURCES_DIR}
        DEPENDS ${headers}
        OUTPUT ${RESOURCES_OBJS})
add_custom_target(unzip_py ALL
        DEPENDS ${RESOURCES_OBJS})

# Project modules
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
if ((CMAKE_VERSION VERSION_EQUAL 3.3) OR (CMAKE_VERSION VERSION_GREATER 3.3))
    cmake_policy(SET CMP0063 NEW)
endif()
add_subdirectory(external/googlebenchmark)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
add_subdirectory(external/cnpy)
add_subdirectory(external/pystring)
add_subdirectory(src/jet)
add_subdirectory(src/tests/manual_tests)
add_subdirectory(src/tests/perf_tests)
add_subdirectory(src/tests/perf_tests2)
add_subdirectory(src/tests/unit_tests)
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_subdirectory(src/winix)
endif()
add_subdirectory(src/examples/hello_fluid_sim)
add_subdirectory(src/examples/hybrid_liquid_sim)
add_subdirectory(src/examples/level_set_liquid_sim)
add_subdirectory(src/examples/obj2sdf)
add_subdirectory(src/examples/particles2obj)
add_subdirectory(src/examples/particles2xml)
add_subdirectory(src/examples/smoke_sim)
add_subdirectory(src/examples/sph_sim)
add_subdirectory(src/examples/playground)

add_subdirectory(external/pybind11)
if (BUILD_FROM_PIP)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()
add_subdirectory(src/python)

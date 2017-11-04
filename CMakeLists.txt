#
# Copyright (c) 2018 Doyub Kim
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

# Options
option(USE_GL "Use OpenGL features" ON)
if (USE_GL)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 REQUIRED)
    if (APPLE)
        find_library(corefoundation_lib CoreFoundation)
    else()
        set(corefoundation_lib "")
    endif()
    set(OPENGL_LIBRARIES ${OPENGL_LIBRARIES}
            ${corefoundation_lib}
            glfw)
    add_definitions(-DJET_USE_GL)

    set(IMGUI_LIBRARIES imgui)
else()
    set(USE_IMGUI OFF CACHE BOOL "" FORCE)
endif()

# Includes
include_directories(/usr/local/include)
include_directories(include)
include_directories(external)
include_directories(external/pybind11/include)
include_directories(external/googletest/googletest/include)
include_directories(external/Clara/include)
include_directories(external/tinyobj)
include_directories(src/examples)
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    include_directories(src/winix)
endif()
if (USE_GL)
    include_directories(external/imgui)
    include_directories(${OPENGL_INCLUDE_DIRS})
else()
    set(OPENGL_LIBRARIES "")
endif()

# Build type - Release by default
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configure tasking system backend
include(cmake/TaskingSystemOptions.cmake)

# Compile options
include(cmake/CompileOptions.cmake)

# Code coverage - Debug only
# NOTE: Code coverage results with an optimized (non-Debug) build may be misleading
if (CMAKE_BUILD_TYPE MATCHES Debug AND CMAKE_COMPILER_IS_GNUCXX)
    include(CodeCoverage)
    setup_target_for_coverage(${PROJECT_NAME}_coverage unit_tests coverage)
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
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

# Project modules
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
if ((CMAKE_VERSION VERSION_EQUAL 3.3) OR (CMAKE_VERSION VERSION_GREATER 3.3))
    cmake_policy(SET CMP0063 NEW)
endif()
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(external/googlebenchmark)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
add_subdirectory(external/cnpy)
add_subdirectory(external/pystring)
if (USE_GL)
    add_subdirectory(external/imgui)
endif()
add_subdirectory(src/jet)
add_subdirectory(src/jet.viz)
add_subdirectory(src/tests/manual_tests)
add_subdirectory(src/tests/mem_perf_tests)
add_subdirectory(src/tests/time_perf_tests)
add_subdirectory(src/tests/unit_tests)
if (USE_GL)
    add_subdirectory(src/tests/opengl_tests)
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
if (USE_GL)
    add_subdirectory(src/examples/smoke_sim2d)
    add_subdirectory(src/examples/stam_sf)
endif()
add_subdirectory(external/pybind11)
if (BUILD_FROM_PIP)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()
add_subdirectory(src/python)

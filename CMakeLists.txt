#
# Copyright (c) 2018 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# CMake version
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Overrides
set(CMAKE_MACOSX_RPATH ON)
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()
if (POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()
if (POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Declare project
project(jet VERSION 1.3.2)

# Configure tasking system backend
include(cmake/TaskingSystemOptions.cmake)

# Common compile options
include(cmake/CompileOptions.cmake)

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Build type - Release by default
message("CMake build type: " ${CMAKE_BUILD_TYPE})
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-DDEBUG)
endif()

# Find Apple-specific libraries
if (APPLE)
    find_library(COCOA_LIB Cocoa)
    find_library(COREFOUNDATION_LIB CoreFoundation)
else()
    set(COCOA_LIB "")
    set(COREFOUNDATION_LIB "")
endif()

# OpenGL
option(USE_GL "Use OpenGL features" ON)
if (USE_GL)
    find_package(OpenGL REQUIRED)
    
    if (OPENGL_FOUND)
        add_definitions(-DJET_USE_GL)

        # Add GLFW3
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            find_program(NUGET nuget)
            if (NOT NUGET)
                message(FATAL "Cannot find nuget command line tool.")
            else()
                if ("${CMAKE_GENERATOR}" MATCHES "Win64" OR "${CMAKE_GENERATOR_PLATFORM}" MATCHES "x64")
                    set(MSVC_ARCH "x64")
                else()
                    set(MSVC_ARCH "Win32")
                endif()
                execute_process(COMMAND ${NUGET} install glfw -Version 3.2.1-v141)
                set(MSVC_GLFW_DIR ${PROJECT_BINARY_DIR}/glfw.3.2.1-v141/build/native)
                include_directories(${MSVC_GLFW_DIR}/include)
                link_directories(${MSVC_GLFW_DIR}/lib/${CMAKE_VS_PLATFORM_TOOLSET}/${MSVC_ARCH}/static)
            endif()
            set(GLFW_LIBRARY glfw3)
        else()
            find_package(glfw3 REQUIRED)
            if (APPLE)
                set(GLFW_LIBRARY glfw)
            else()
                # From pkg-config glfw3 --static --libs
                set(GLFW_LIBRARY
                    glfw
                    rt m dl Xrandr Xinerama Xcursor Xext Xrender Xfixes X11 pthread
                    xcb Xau Xdmcp)
            endif()
        endif()

        set(OPENGL_LIBRARIES 
            ${OPENGL_LIBRARIES}
            ${GLFW_LIBRARY}
            ${COREFOUNDATION_LIB})
    else()
        set(USE_GL OFF)
    endif()
endif()

# CUDA
option(USE_CUDA "Use CUDA features" ON)
if (USE_CUDA)
    find_package(CUDA)

    if (CUDA_FOUND)
        set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
        list(APPEND CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets)
        add_definitions(-DJET_USE_CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})

        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
                /wd4515 # -> disable warning: [Thrust issue] 'detail': namespace uses itself
                /wd4996 # -> disable warning: [Thrust issue] 'strerror': This function or variable may be unsafe.
                /wd4800 # -> disable warning: [Thrust issue] forcing value to bool 'true' or 'false' (performance warning)

                # Temp
                /wd4505
            )
        else()
            list(APPEND CUDA_NVCC_FLAGS --std=c++14)
        endif()
    else()
        set(USE_CUDA OFF)
    endif()
endif()

# Include directories
if (WIN32)
    set(USR_LOCAL_INCLUDE_DIR "")
else()
    set(USR_LOCAL_INCLUDE_DIR /usr/local/include)
endif()
set(JET_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(EXTERNAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(PYBIND11_INCLUDE_DIR ${EXTERNAL_INCLUDE_DIR}/pybind11/include)
set(GTEST_INCLUDE_DIR ${EXTERNAL_INCLUDE_DIR}/googletest/googletest/include)
set(CLARA_INCLUDE_DIR ${EXTERNAL_INCLUDE_DIR}/Clara/include)
set(TINYOBJ_INCLUDE_DIR ${EXTERNAL_INCLUDE_DIR}/tinyobj)
set(EXAMPLES_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/examples
    ${EXTERNAL_INCLUDE_DIR}
    ${CLARA_INCLUDE_DIR})

# Build type - Release by default
message("CMake build type: " ${CMAKE_BUILD_TYPE})
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-DDEBUG)
endif()

# Configure tasking system backend
include(cmake/TaskingSystemOptions.cmake)

# Compile options
include(cmake/CompileOptions.cmake)

# Code coverage - Debug only
# NOTE: Code coverage results with an optimized (non-Debug) build may be misleading
option(BUILD_COVERAGE "Build code coverage" OFF)
if (CMAKE_BUILD_TYPE MATCHES Debug AND CMAKE_COMPILER_IS_GNUCXX AND BUILD_COVERAGE)
    include(CodeCoverage)
    setup_target_for_coverage(${PROJECT_NAME}_coverage unit_tests coverage)
endif()

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
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(external/googlebenchmark)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
add_subdirectory(external/cnpy)
add_subdirectory(external/pystring)
# if (USE_GL)
#     add_subdirectory(external/imgui)
# endif()
add_subdirectory(src/jet)
add_subdirectory(src/jet.gfx)
add_subdirectory(src/tests/jtl_tests)
add_subdirectory(src/tests/manual_tests)
add_subdirectory(src/tests/mem_perf_tests)
add_subdirectory(src/tests/time_perf_tests)
add_subdirectory(src/tests/unit_tests)
# if (USE_GL)
#     add_subdirectory(src/tests/opengl_tests)
# endif()
# add_subdirectory(src/examples/common)
add_subdirectory(src/examples/hello_fluid_sim)
add_subdirectory(src/examples/hybrid_liquid_sim)
add_subdirectory(src/examples/level_set_liquid_sim)
add_subdirectory(src/examples/obj2sdf)
add_subdirectory(src/examples/particles2obj)
add_subdirectory(src/examples/particles2xml)
add_subdirectory(src/examples/smoke_sim)
add_subdirectory(src/examples/sph_sim)
add_subdirectory(src/examples/playground)
add_subdirectory(src/examples/example_utils)
if (USE_GL)
    add_subdirectory(src/examples/opengl_examples)
endif()
if (APPLE)
    add_subdirectory(src/examples/metal_examples)
endif()
# add_subdirectory(src/examples/particle_sim)
# if (USE_GL)
#     add_subdirectory(src/examples/smoke_sim2d)
# endif()
add_subdirectory(external/pybind11)
if (BUILD_FROM_PIP)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()
add_subdirectory(src/python)

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

# Declare project
project(jet)

# Overrides
set(CMAKE_MACOSX_RPATH ON)
cmake_policy(SET CMP0023 OLD) # TODO: blaming FindCUDA in default CMake for Ubuntu 14.04
if ((CMAKE_VERSION VERSION_EQUAL 3.3) OR (CMAKE_VERSION VERSION_GREATER 3.3))
    cmake_policy(SET CMP0063 NEW)
endif()

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
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenGL
option(USE_GL "Use OpenGL features" ON)
if (USE_GL)
    find_package(OpenGL REQUIRED)

    # Add Apple-specific libraries
    if (APPLE)
        find_library(COREFOUNDATION_LIB CoreFoundation)
    else()
        set(COREFOUNDATION_LIB "")
    endif()

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

    set(OPENGL_LIBRARIES ${OPENGL_LIBRARIES}
            ${COREFOUNDATION_LIB}
            ${GLFW_LIBRARY})
    add_definitions(-DJET_USE_GL)

    # Add dear ImGui
    set(IMGUI_LIBRARIES imgui)

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
    set(IMGUI_LIBRARIES "")
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

# Options
# CUDA
option(USE_CUDA "Use CUDA features" ON)
if (USE_CUDA)
    find_package(CUDA)

    if (CUDA_FOUND)
        list(APPEND CUDA_NVCC_FLAGS
             -Wno-deprecated-gpu-targets
                -arch=sm_30
                -gencode=arch=compute_30,code=sm_30
                -gencode=arch=compute_50,code=sm_50
                -gencode=arch=compute_52,code=sm_52
                -gencode=arch=compute_60,code=sm_60
                -gencode=arch=compute_61,code=sm_61
                -gencode=arch=compute_61,code=compute_61)
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
            list(APPEND CUDA_NVCC_FLAGS --std=c++11)
        endif()
    else()
        set(USE_CUDA OFF)
    endif()
endif()

# Includes
if (UNIX)
    include_directories(/usr/local/include)
endif()
include_directories(include)
include_directories(external)
include_directories(external/pybind11/include)
include_directories(external/googletest/googletest/include)
include_directories(external/googlebenchmark/include)
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    include_directories(src/winix)
endif()

# Build type - Release by default
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
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
add_subdirectory(src/examples/common)
add_subdirectory(src/examples/hello_fluid_sim)
add_subdirectory(src/examples/hybrid_liquid_sim)
add_subdirectory(src/examples/level_set_liquid_sim)
add_subdirectory(src/examples/obj2sdf)
add_subdirectory(src/examples/particles2obj)
add_subdirectory(src/examples/particles2xml)
add_subdirectory(src/examples/smoke_sim)
add_subdirectory(src/examples/sph_sim)
add_subdirectory(src/examples/playground)
add_subdirectory(src/examples/particle_sim)
if (USE_GL)
    add_subdirectory(src/examples/smoke_sim2d)
    add_subdirectory(src/examples/stam_sf)
endif()
add_subdirectory(external/pybind11)
if (BUILD_FROM_PIP)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endif()
add_subdirectory(src/python)

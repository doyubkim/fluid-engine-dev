#
# Copyright (c) 2018 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# Target name
set(target time_perf_tests)

# Includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(external/googlebenchmark/include)

# Sources
if (ENABLE_CUDA)
    file(GLOB sources
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
else()
    file(GLOB sources
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
endif()

# Build executable
if (ENABLE_CUDA)
    cuda_add_executable(${target}
        ${sources})
else()
    add_executable(${target}
        ${sources})
endif()

# Project options
set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS})

# Compile options
target_compile_options(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}

        INTERFACE)

target_compile_definitions(${target}
        PRIVATE
        RESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../resources/")

# Link libraries
target_link_libraries(${target}
        ${DEFAULT_LINKER_OPTIONS}
        jet
        benchmark)

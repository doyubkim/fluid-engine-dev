#
# Copyright (c) 2018 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# Target name
set(target examples_common)

# Includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../jet)

# Sources
file(GLOB sources
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build library
add_library(${target} ${sources})

# Project options
set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS})

# Compile options
target_compile_options(${target}
        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS})

target_link_libraries(${target}
        PUBLIC
        ${DEFAULT_LINKER_OPTIONS})

# Link libraries
target_link_libraries(${target}
        ${DEFAULT_LINKER_OPTIONS}
        jet.viz
        jet
        ${IMGUI_LIBRARIES}
        ${OPENGL_LIBRARIES})

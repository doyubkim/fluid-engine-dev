#
# Copyright (c) 2018 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# Target name
set(target example_utils)

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
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS})

# Includes
target_include_directories(${target}
    PRIVATE
    ${JET_INCLUDE_DIR}
    ${EXAMPLES_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
    jet
    jet.gfx
    ${OPENGL_LIBRARIES}
)

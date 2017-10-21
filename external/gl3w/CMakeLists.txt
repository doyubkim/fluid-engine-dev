#
# Copyright (c) 2017 Doyub Kim
#

# Target name
set(target gl3w)

# Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Sources
file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/GL/gl3w.c
)

# Build library
add_library(${target}
    ${sources}
)

# Project options
set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
)

# Compile options
target_compile_options(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS}

    INTERFACE
)

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
)

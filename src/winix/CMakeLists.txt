#
# Copyright (c) 2017 Doyub Kim
#

# Target name
set(target winix)

# Sources
set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/getopt_long.c
    ${CMAKE_CURRENT_SOURCE_DIR}/getopt.c
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
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
        /wd4131       # -> disable warning: 'xxx': uses old-style declarator
        /wd4244       # -> disable warning: 'initializing': conversion from 'unsigned __int64' to 'unsigned int', possible loss of data
        /wd4267       # -> disable warning: '=': conversion from 'size_t' to 'int', possible loss of data
        /wd4706       # -> disable warning: assignment within conditional expression
        )
endif ()

target_compile_options(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS}

    INTERFACE
)

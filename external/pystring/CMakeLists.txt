#
# Copyright (c) 2018 Doyub Kim
#

# Target name
set(target pystring)

# Sources
set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/pystring.cpp
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
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
        /wd4800       # -> disable warning: 'int': forcing value to bool 'true' or 'false' (performance warning)
        )
endif ()

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

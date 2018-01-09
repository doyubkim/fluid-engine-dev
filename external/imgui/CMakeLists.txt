#
# Copyright (c) 2017 Doyub Kim
#

# Target name
set(target imgui)

# Includes
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR})

# Sources
file(GLOB sources
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GL/*.cpp)

# Build library
add_library(${target}
        ${sources})

# Project options
set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS})

# Compile options
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(IMGUI_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
            -Wno-maybe-uninitialized)
else()
    set(IMGUI_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS})
endif()

target_compile_options(${target}
        PRIVATE

        PUBLIC
        ${IMGUI_COMPILE_OPTIONS}

        INTERFACE)

target_link_libraries(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}

        INTERFACE)

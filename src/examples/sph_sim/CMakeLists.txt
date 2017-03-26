#
# Copyright (c) 2017 Doyub Kim
#

# Target name
set(target sph_sim)

# Includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Sources
file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build executable
add_executable(${target}
    ${sources})

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
target_compile_definitions(${target}
    PRIVATE
    RESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../resources/"
)

# Link libraries
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    target_link_libraries(${target}
        jet
        pystring
        winix)
else()
    target_link_libraries(${target}
        jet
        pystring)
endif()

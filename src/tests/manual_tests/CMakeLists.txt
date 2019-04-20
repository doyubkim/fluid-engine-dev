#
# Copyright (c) 2019 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# Target name
set(target manual_tests)

# Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EXTERNAL_DIR}/googletest/googlemock/include
    ${EXTERNAL_DIR}/googletest/googletest/include)

# Sources
file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build executable
add_executable(${target}
    ${sources})

# Dependencies
add_dependencies(${target} unzip_py)

# Project options
set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
)

# Compile options
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(DEFAULT_COMPILE_OPTIONS ${DEFAULT_COMPILE_OPTIONS}
        /wd4018       # -> disable warning: signed/unsigned mismatch (because of cnpy)
        /wd4100       # -> disable warning: unreferenced formal parameter (because of cnpy)
        /wd4310       # -> disable warning: cast truncates constant value (because of cnpy)
        )
    add_definitions(-D_USE_MATH_DEFINES) # for M_PI
endif ()

target_compile_options(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS}

    INTERFACE
)
target_compile_definitions(${target}
    PRIVATE
    RESOURCES_DIR="${RESOURCES_DIR}"
)

# Link libraries
target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
    jet
    cnpy
    gtest
    pystring)

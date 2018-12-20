#
# Copyright (c) 2018 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# Target name
set(target mem_perf_tests)

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
    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS}
)
target_compile_definitions(${target}
    PRIVATE
    RESOURCES_DIR="${RESOURCES_DIR}"
)

# Includes
target_include_directories(${target}
    PRIVATE
    ${JET_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(${target}
    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}
    jet
    gtest)

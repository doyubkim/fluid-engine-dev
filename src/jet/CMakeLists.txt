#
# Copyright (c) 2018 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# Target name
set(target jet)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
)

# Sources
file(GLOB header_dir
    ${root_dir}/include/${target})

file(GLOB headers
    ${header_dir}/*.h
    ${header_dir}/detail/*.h)
list(REMOVE_ITEM headers ${header_dir}/jet.h)

if (USE_CUDA)
    file(GLOB sources
        ${headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/obj/*.cpp)
else()
    file(GLOB sources
        ${headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/obj/*.cpp)
endif()

# Custom-build event
set(jet_header_gen_py ${root_dir}/scripts/header_gen.py)
add_custom_command(
    COMMAND python ${jet_header_gen_py}
    DEPENDS ${headers} ${root_dir}/scripts/header_gen.py
    OUTPUT ${root_dir}/include/jet/jet.h
)
add_custom_target(jet_header_gen_py ALL
    DEPENDS ${root_dir}/include/jet/jet.h)

# Build library
if (USE_CUDA)
    cuda_add_library(${target}
        ${sources})
else()
    add_library(${target}
        ${sources})
endif()

add_dependencies(${target} jet_header_gen_py)

# Project options
set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    CUDA_SEPARABLE_COMPILATION ON
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
    ${DEFAULT_LIBRARIES}

    INTERFACE
)

# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include)

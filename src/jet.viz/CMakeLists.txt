#
# Copyright (c) 2017 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# Target name
set(target jet.viz)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Includes
if (USE_GL)
    include_directories(
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gl3w)
    if (USE_CUDA)
        include_directories(
                ${CUDA_INCLUDE_DIRECTORIES}
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gl3w)
    endif()
else()
    include_directories(
            ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Sources
file(GLOB header_dir
        ${root_dir}/include/${target})

file(GLOB headers
        ${header_dir}/*.h
        ${header_dir}/detail/*.h)
list(REMOVE_ITEM headers ${header_dir}/jet.viz.h)

file(GLOB sources
        ${headers}
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if (USE_GL)
    file(GLOB sources
            ${sources}
            ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gl3w/GL/gl3w.cpp)
endif()
if (USE_CUDA)
    file(GLOB sources
            ${sources}
            ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
endif()

# Custom-build event
set(jet_viz_header_gen_py ${root_dir}/scripts/header_gen.py)
add_custom_command(
        COMMAND python ${jet_viz_header_gen_py} jet.viz
        DEPENDS ${headers} ${root_dir}/scripts/header_gen.py
        OUTPUT ${root_dir}/include/jet.viz/jet.viz.h)
add_custom_target(jet_viz_header_gen_py ALL
        DEPENDS ${root_dir}/include/jet.viz/jet.viz.h)

# Build library
if (USE_CUDA)
    cuda_add_library(${target}
        ${sources})
else()
    add_library(${target}
        ${sources})
endif()

add_dependencies(${target} jet_viz_header_gen_py)

# Project options
set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS})

# Compile options
target_compile_options(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}

        INTERFACE)

target_link_libraries(${target}
        ${DEFAULT_LINKER_OPTIONS})

# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include)

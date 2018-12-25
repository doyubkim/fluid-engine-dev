#
# Copyright (c) 2018 Doyub Kim
#
# I am making my contributions/submissions to this project solely in my personal
# capacity and am not conveying any rights to any intellectual property of any
# third parties.
#

# Target name
set(target jet.gfx)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)

set(JET_GFX_INCLUDE_DIRS
    ${JET_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

if (USE_GL)
    list(APPEND JET_GFX_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gl3w)
endif()
if (USE_CUDA)
list(APPEND JET_GFX_INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRECTORIES})
endif()

# Sources
file(GLOB header_dir
    ${root_dir}/include/${target})

file(GLOB headers
    ${header_dir}/*.h
    ${header_dir}/detail/*.h)
list(REMOVE_ITEM headers ${header_dir}/jet.gfx.h)

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
if (APPLE)
    file(GLOB sources
        ${sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/*.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mtlpp/*.mm)
endif()

# Custom-build event
set(jet_gfx_header_gen_py ${root_dir}/scripts/header_gen.py)
add_custom_command(
    COMMAND python ${jet_gfx_header_gen_py} jet.gfx
    DEPENDS ${headers} ${root_dir}/scripts/header_gen.py
    OUTPUT ${root_dir}/include/jet.gfx/jet.gfx.h)
add_custom_target(jet_gfx_header_gen_py ALL
    DEPENDS ${root_dir}/include/jet.gfx/jet.gfx.h)

# Build library
if (USE_CUDA)
    cuda_add_library(${target}
    ${sources})
else()
    add_library(${target}
    ${sources})
endif()

add_dependencies(${target} jet_gfx_header_gen_py)

# Project options
set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS})

# Compile options
target_compile_options(${target}
    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS})

target_link_libraries(${target}
    PUBLIC
    ${DEFAULT_LINKER_OPTIONS})

# Includes
target_include_directories(${target}
    PRIVATE
    ${JET_GFX_INCLUDE_DIRS}

    PUBLIC

    INTERFACE
    $<BUILD_INTERFACE:${header_dir}>
    $<INSTALL_INTERFACE:include>
)

# Install
install(TARGETS ${target} DESTINATION lib EXPORT ${target}_Export)
install(DIRECTORY ${header_dir} DESTINATION include)
install(EXPORT ${target}_Export
    DESTINATION lib/cmake/${target}
    FILE ${target}Config.cmake
)

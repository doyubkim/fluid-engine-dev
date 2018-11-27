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

# Sources
file(GLOB header_dir
    ${root_dir}/include/${target})

file(GLOB headers
    ${header_dir}/*.h
    ${header_dir}/detail/*.h)

file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Custom-build event
set(jet_header_gen_py ${root_dir}/scripts/header_gen.py)
add_custom_command(
    COMMAND python ${jet_header_gen_py}
    DEPENDS ${headers}
    OUTPUT ${root_dir}/include/jet.h
)
add_custom_target(jet_header_gen_py ALL
    DEPENDS ${root_dir}/include/jet.h)

# Build library
add_library(${target}
    ${sources})
add_dependencies(${target} jet_header_gen_py)

# Project options
set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
)

# Compile options
target_compile_options(${target}
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}

    PUBLIC

    INTERFACE
)

target_link_libraries(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}
    ${DEFAULT_LIBRARIES}

    INTERFACE
)

# Includes
target_include_directories(${target}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty

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

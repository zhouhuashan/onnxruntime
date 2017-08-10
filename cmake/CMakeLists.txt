# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(CommonIR C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
set(COMMONIR_ROOT ${PROJECT_SOURCE_DIR}/..)
include(zlib)
include(protobuf)

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${PROTOBUF_INCLUDE_DIRS}
)

include(commonIR_core_protobuf.cmake)
include(commonIR_core_graph.cmake)
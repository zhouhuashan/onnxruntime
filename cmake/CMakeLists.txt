# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(LotusIR C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
set(LOTUSIR_ROOT ${PROJECT_SOURCE_DIR}/..)
include(zlib)
include(protobuf)
include(googletest)

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${PROTOBUF_INCLUDE_DIRS}
)

include(lotusIR_core_protobuf.cmake)
#include(onnx_core_protobuf.cmake)
include(lotusIR_core_graph.cmake)
include(lotusIR_unittest.cmake)

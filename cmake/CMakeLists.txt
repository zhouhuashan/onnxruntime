# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(LotusIR C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# Options
option(lotusIR_RUN_ONNX_TESTS "Enable ONNX Compatibility Testing" OFF)

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
set(LOTUSIR_ROOT ${PROJECT_SOURCE_DIR}/..)
include(zlib)
include(protobuf)
include(googletest)
include(models)
if (NOT lotusIR_RUN_ONNX_TESTS)
  set_target_properties(models PROPERTIES EXCLUDE_FROM_ALL 1)
else()
  add_definitions(-DLOTUSIR_RUN_EXTERNAL_ONNX_TESTS)
endif()

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${PROTOBUF_INCLUDE_DIRS}
)

add_definitions(-DONNX_V1_OPSCHEMA_COMPAT)
include(lotusIR_core_protobuf.cmake)
#include(onnx_core_protobuf.cmake)
include(lotusIR_core_graph.cmake)
include(lotusIR_unittest.cmake)

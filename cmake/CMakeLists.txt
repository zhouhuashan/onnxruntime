# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(LotusIR C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# Enable CTest
enable_testing()

# Options
option(lotusIR_RUN_ONNX_TESTS "Enable ONNX Compatibility Testing" OFF)
option(lotusIR_GENERATE_TEST_REPORTS "Enable test report generation" OFF)

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
set(LOTUSIR_ROOT ${PROJECT_SOURCE_DIR}/..)
include(zlib)
include(protobuf)
include(googletest)
if (lotusIR_RUN_ONNX_TESTS)
  include(models)
  add_definitions(-DLOTUSIR_RUN_EXTERNAL_ONNX_TESTS)
endif()

if (WIN32)
    # parallel build
    add_compile_options(/MP)

    # compile warning level 4
    add_compile_options(/W4)

    # treat warnings as errors
    add_compile_options(/WX)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WX")
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /WX")
endif()

include_directories(
    ${LOTUSIR_ROOT}
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${PROTOBUF_INCLUDE_DIRS}
)

add_definitions(-DONNX_V1_OPSCHEMA_COMPAT)
include(lotusIR_util.cmake)
include(lotusIR_core_protobuf.cmake)
#include(onnx_core_protobuf.cmake)
include(lotusIR_core_graph.cmake)
include(lotusIR_unittest.cmake)

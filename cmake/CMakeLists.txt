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

# Python3
if (NOT DEFINED PYTHON3_EXECUTABLE)
    find_package(PythonInterp 3)
    if (NOT PYTHONINTERP_FOUND)
        message(FATAL_ERROR "Failed to find Python3!")
    endif()
    set(PYTHON3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

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
    # warning level 4
    add_compile_options(/W4)
endif()

include_directories(
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

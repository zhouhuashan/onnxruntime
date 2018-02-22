# Minimum CMake required
cmake_minimum_required(VERSION 3.10)

# Project
project(Lotus C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# Enable CTest
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo." FORCE)
endif()

# Options
option(lotus_RUN_ONNX_TESTS "Enable ONNX Compatibility Testing" OFF)
option(lotus_GENERATE_TEST_REPORTS "Enable test report generation" OFF)
option(lotus_ENABLE_STATIC_ANALYSIS "Enable static analysis" OFF)
option(lotus_ENABLE_STDERR_LOGGING "Enable logging to stderr" ON)
if(CMAKE_BUILD_TYPE MATCHES Debug)
option(lotus_ENABLE_DEBUG_LOGGING "Enable debug logging (e.g., enable VLOG)" ON)
else()
option(lotus_ENABLE_DEBUG_LOGGING "Enable debug logging (e.g., enable VLOG)" OFF)
endif()

include(lotusIR_util.cmake)
# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
set(LOTUS_ROOT ${PROJECT_SOURCE_DIR}/../lotus)
include(zlib)
include(protobuf)
include(googletest)
include(eigen)
if (lotus_RUN_ONNX_TESTS)
  include(models)
  add_definitions(-DLOTUSIR_RUN_EXTERNAL_ONNX_TESTS)
endif()
include(onnx.cmake)
add_definitions(-DONNX_ML)

set(lotus_EXTERNAL_DEPENDENCIES
    eigen
)

if (lotus_ENABLE_DEBUG_LOGGING)
  add_definitions(-DLOTUS_ENABLE_DEBUG_LOGGING)
else()
  add_definitions(-DLOTUS_LOG_THRESHOLD=2)
endif()

if (lotus_ENABLE_STDERR_LOGGING)
  add_definitions(-DLOTUS_ENABLE_STDERR_LOGGING)
endif()

if (WIN32)
    add_definitions(-DPLATFORM_WINDOWS)
    # parallel build
    add_compile_options(/MP)
    add_compile_options(
      /wd4503 # Decorated name length exceeded.
      /wd4127 # conditional expression is constant
    )
    if (lotus_ENABLE_STATIC_ANALYSIS)
        add_compile_options(
            /analyze:WX-
            # disable warning because there are many occurrences from test macros
            /wd6326 # potential comparison of a constant with another constant
        )
    endif()

    # compile warning level 4
    add_compile_options(/W4)

    # treat warnings as errors
    add_compile_options(/WX)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WX")
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /WX")
else()
    add_definitions(-DPLATFORM_POSIX)    
endif()

include_directories(
    ${LOTUS_ROOT}
    ${PROJECT_SOURCE_DIR}/../external/onnx
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/external/onnx
    # External dependencies.
    ${PROTOBUF_INCLUDE_DIRS}
    ${eigen_INCLUDE_DIRS}
)
include(lotus_common.cmake)
include(lotusIR_core_graph.cmake)
include(lotus_core_framework.cmake)
include(lotusIR_unittest.cmake)

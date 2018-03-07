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
option(lotus_USE_CUDA "Build with CUDA support" OFF)
option(lotus_USE_EIGEN_FOR_BLAS "Use eign for blas" ON)
option(lotus_USE_OPENBLAS "Use openblas" OFF)
if(CMAKE_BUILD_TYPE MATCHES Debug)
option(lotus_ENABLE_DEBUG_LOGGING "Enable debug logging (e.g., enable VLOG)" ON)
else()
option(lotus_ENABLE_DEBUG_LOGGING "Enable debug logging (e.g., enable VLOG)" OFF)
endif()

if (lotus_USE_CUDA AND "${lotus_CUDNN_HOME}" STREQUAL "")
  message(FATAL_ERROR "lotus_CUDNN_HOME required for lotus_USE_CUDA")
endif()

if (lotus_USE_EIGEN_FOR_BLAS)
add_definitions(-DLOTUS_USE_EIGEN_FOR_BLAS)
endif()

if (lotus_USE_OPENBLAS AND "${lotus_OPENBLAS_HOME}" STREQUAL "")
  message(FATAL_ERROR "lotus_OPENBLAS_HOME required for lotus_USE_OPENBLAS")
endif()

if (lotus_USE_OPENBLAS AND lotus_USE_EIGEN_FOR_BLAS)
  message(FATAL_ERROR "use one of lotus_USE_OPENBLAS, lotus_USE_EIGEN_FOR_BLAS")
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
    if (NOT lotus_USE_CUDA)
      add_compile_options(/WX)
      foreach(type EXE STATIC SHARED)
        set(CMAKE_${type}_LINKER_FLAGS "${CMAKE_${type}_LINKER_FLAGS} /WX")
      endforeach()
    endif()
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

set(lotus_EXTERNAL_LIBRARIES
    ${zlib_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
)

if (lotus_USE_OPENBLAS)
  add_definitions(-DUSE_OPENBLAS=1)
  if (WIN32)
    include_directories(${lotus_OPENBLAS_HOME})
    list(APPEND lotus_EXTERNAL_LIBRARIES ${lotus_OPENBLAS_HOME}/lib/libopenblas.lib)
  else()
    # on linux we assume blas is installed via 'apt install libopenblas-dev'
    list(APPEND lotus_EXTERNAL_LIBRARIES openblas)
  endif()
endif()

if (lotus_USE_CUDA)
    find_package(CUDA 8.0 REQUIRED)
    include(cub)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=\"sm_35,compute_35\";-gencode arch=compute_52,code=\"sm_52,compute_52\")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--include-path ${PROJECT_BINARY_DIR}/$\{build_configuration\};--expt-relaxed-constexpr)
    if (WIN32)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)
    else()
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true;-std=c++11;-Xcompiler -fPIC)
    endif()

    set(CUDA_INCLUDE ${CUDA_TOOLKIT_TARGET_DIR} ${CUDA_TOOLKIT_TARGET_DIR}/extras/CUPTI/include
      ${lotus_CUDNN_HOME}/include ${cub_INCLUDE_DIR})
    include_directories(${CUDA_INCLUDE})
    add_definitions(-DUSE_CUDA=1)

    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUFFT_LIBRARIES}
      ${CUDA_curand_LIBRARY} ${CUDA_cupti_LIBRARY} ${CUDA_cusolver_LIBRARY})
    if (WIN32)
      list(APPEND CUDA_LIBRARIES ${lotus_CUDNN_HOME}/lib/x64/cudnn.lib)
    else()
      list(APPEND CUDA_LIBRARIES ${lotus_CUDNN_HOME}/lib64/libcudnn.so ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libculibos.a)
    endif()
    list(APPEND lotus_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES})
    list(APPEND lotus_EXTERNAL_DEPENDENCIES cub)
endif()

include(lotus_common.cmake)
include(lotus_core_framework.cmake)
include(lotus_kernels.cmake)
include(lotusIR_core_graph.cmake)
include(lotus_providers.cmake)
include(lotusIR_unittest.cmake)

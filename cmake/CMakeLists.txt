# Minimum CMake required
cmake_minimum_required(VERSION 3.10)

# Project
project(Lotus C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

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
option(lotus_ENABLE_PYTHON "Enable python buildings" OFF)
option(lotus_USE_CUDA "Build with CUDA support" OFF)
option(lotus_USE_EIGEN_FOR_BLAS "Use eign for blas" ON)
option(lotus_USE_OPENBLAS "Use openblas" OFF)
option(lotus_USE_PREBUILT_PB "Use prebuilt protobuf library" OFF)
option(lotus_MSVC_STATIC_RUNTIME "Compile for the static CRT" OFF)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
set(ONNX_ML 1)

if (MSVC)
  if (lotus_MSVC_STATIC_RUNTIME)
    # set all of our submodules to static runtime
    set(ONNX_USE_MSVC_STATIC_RUNTIME ON)
    set(protobuf_MSVC_STATIC_RUNTIME ON)
    set(gtest_force_shared_crt OFF)

    # In case we are building static libraries, link also the runtime library statically
    # so that MSVCR*.DLL is not required at runtime.
    # https://msdn.microsoft.com/en-us/library/2kzt1wy3.aspx
    # This is achieved by replacing msvc option /MD with /MT and /MDd with /MTd
    # http://www.cmake.org/Wiki/CMake_FAQ#How_can_I_build_my_MSVC_application_with_a_static_runtime.3F  
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE__FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  else()
    set(ONNX_USE_MSVC_STATIC_RUNTIME OFF)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Link protobuf to static runtime libraries" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib for gtest" FORCE)
  endif()
endif()


#Here we support three build mode:
#1. (recommended)lotus_USE_PREBUILT_PB is set (ONNX_CUSTOM_PROTOC_EXECUTABLE should also be set)
#   We will not build protobuf, will use a prebuilt binary instead. This mode can also support cross-compiling
#2. lotus_USE_PREBUILT_PB is not set but ONNX_CUSTOM_PROTOC_EXECUTABLE is set
#   Build Protobuf from source, except protoc.exe. This mode is mainly for cross-compiling
#3. both lotus_USE_PREBUILT_PB and ONNX_CUSTOM_PROTOC_EXECUTABLE are not set
#   Compile everything from source code. Slowest option.

if(lotus_USE_PREBUILT_PB)
  get_filename_component(
     _PROTOBUF_INSTALL_PREFIX
     ${ONNX_CUSTOM_PROTOC_EXECUTABLE}
     DIRECTORY)
  get_filename_component(
    _PROTOBUF_INSTALL_PREFIX
    ${_PROTOBUF_INSTALL_PREFIX}/..
    REALPATH)
  include(${_PROTOBUF_INSTALL_PREFIX}/cmake/protobuf-config.cmake)
else()
  # use protobuf as a submodule
  add_subdirectory(${PROJECT_SOURCE_DIR}/external/protobuf/cmake EXCLUDE_FROM_ALL)
  set_target_properties(libprotobuf PROPERTIES FOLDER "External/Protobuf")
  set_target_properties(libprotobuf-lite PROPERTIES FOLDER "External/Protobuf")
  set_target_properties(libprotoc PROPERTIES FOLDER "External/Protobuf")
  set_target_properties(protoc PROPERTIES FOLDER "External/Protobuf")
  set_target_properties(js_embed PROPERTIES FOLDER "External/Protobuf")
  add_library(protobuf::libprotobuf ALIAS libprotobuf)
  add_executable(protobuf::protoc ALIAS protoc)
endif()
get_target_property(PROTOBUF_INCLUDE_DIRS protobuf::libprotobuf
    INTERFACE_INCLUDE_DIRECTORIES)

set(googletest_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/googletest/googletest/include ${PROJECT_SOURCE_DIR}/external/googletest/googlemock/include)

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


if(WIN32)
  #Always enable exception handling, even for Windows ARM
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
set(LOTUS_ROOT ${PROJECT_SOURCE_DIR}/../lotus)

include(date)
include(gsl)
include(eigen)
set(lotus_EXTERNAL_DEPENDENCIES
    eigen
)

# zlib
include(zlib)
set_target_properties(zlib_copy_headers_to_destination PROPERTIES FOLDER "External/zlib")
set_target_properties(zlib_create_destination_dir PROPERTIES FOLDER "External/zlib")

# gtest and gmock
add_subdirectory(${PROJECT_SOURCE_DIR}/external/googletest EXCLUDE_FROM_ALL)
set_target_properties(gmock PROPERTIES FOLDER "External/GTest")
set_target_properties(gmock_main PROPERTIES FOLDER "External/GTest")
set_target_properties(gtest PROPERTIES FOLDER "External/GTest")
set_target_properties(gtest_main PROPERTIES FOLDER "External/GTest")

# ONNX
add_subdirectory(${PROJECT_SOURCE_DIR}/external/onnx EXCLUDE_FROM_ALL)
set_target_properties(onnx PROPERTIES FOLDER "External/ONNX")
set_target_properties(onnx_proto PROPERTIES FOLDER "External/ONNX")
add_definitions(-DONNX_ML)
add_definitions(-DONNX_NAMESPACE=onnx)


if (lotus_RUN_ONNX_TESTS)
  include(models)
  add_definitions(-DLOTUSIR_RUN_EXTERNAL_ONNX_TESTS)
endif()

if (WIN32)
    add_definitions(-DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES)
    # parallel build
    # These compiler opitions cannot be forwarded to NVCC, so cannot use add_compiler_options
    string(APPEND CMAKE_CXX_FLAGS " /MP")
    string(APPEND CMAKE_CXX_FLAGS 
      " /wd4503" # Decorated name length exceeded.
      " /wd4127" # conditional expression is constant. 
      " /wd4146" # unary minus operator applied to unsigned type. Needed for Protobuf
    )
    if (lotus_ENABLE_STATIC_ANALYSIS)
        string(APPEND CMAKE_CXX_FLAGS 
            " /analyze:WX- "
            # disable warning because there are many occurrences from test macros
            " /wd6326 " # potential comparison of a constant with another constant
        )
    endif()

    # compile warning level 4
    string(APPEND CMAKE_CXX_FLAGS " /W4")  

    if (NOT lotus_USE_CUDA AND CMAKE_SIZEOF_VOID_P EQUAL 8)
      # treat warnings as errors
      add_compile_options(/WX)
      foreach(type EXE STATIC SHARED)
        set(CMAKE_${type}_LINKER_FLAGS "${CMAKE_${type}_LINKER_FLAGS} /WX")
      endforeach()
    endif()
else()
    add_definitions(-DPLATFORM_POSIX)    
    add_compile_options(-Werror)
endif()

include_directories(
    ${LOTUS_ROOT}
    ${PROJECT_SOURCE_DIR}/external/onnx
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/external/onnx
    # External dependencies.
    ${PROTOBUF_INCLUDE_DIRS}
    ${eigen_INCLUDE_DIRS}
    ${date_INCLUDE_DIR}
    ${gsl_INCLUDE_DIR}
)

set(lotus_EXTERNAL_LIBRARIES protobuf::libprotobuf)

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
    enable_language(CUDA)
    find_package(CUDA 9.0)
    include(cub)
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
include(lotusIR_graph.cmake)
include(lotus_framework.cmake)
include(lotus_providers.cmake)
include(lotus_unittests.cmake)

if (lotus_ENABLE_PYTHON)
if(UNIX)
  set(CMAKE_SKIP_BUILD_RPATH ON)
endif()
include(lotus_python.cmake)
endif()
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# internal providers
# this file is meant to be included in the parent onnxruntime_providers.cmake file

if (onnxruntime_USE_NUPHAR)
  add_definitions(-DUSE_NUPHAR=1)
  
  if (NOT onnxruntime_USE_TVM)
    message(FATAL_ERROR "onnxruntime_USE_TVM required for onnxruntime_USE_NUPHAR")
  endif()

  file(GLOB_RECURSE onnxruntime_providers_nuphar_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/nuphar/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/nuphar/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_nuphar_cc_srcs})
  add_library(onnxruntime_providers_nuphar ${onnxruntime_providers_nuphar_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_nuphar onnx protobuf::libprotobuf)
  set_target_properties(onnxruntime_providers_nuphar PROPERTIES FOLDER "Lotus")
  target_include_directories(onnxruntime_providers_nuphar PRIVATE ${ONNXRUNTIME_ROOT} ${TVM_INCLUDES})
  set_target_properties(onnxruntime_providers_nuphar PROPERTIES LINKER_LANGUAGE CXX)
  target_compile_options(onnxruntime_providers_nuphar PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
  add_dependencies(onnxruntime_providers_nuphar ${onnxruntime_tvm_dependencies})

  # use this if you want this provider to be included in the onnxruntime shared library
  list(APPEND onnxruntime_libs onnxruntime_providers_nuphar)

endif()


if (onnxruntime_USE_BRAINSLICE)
  add_definitions(-DUSE_BRAINSLICE=1)
  if (WIN32)
    include_directories(${lotus_FPGA_CORE_LIB_PATH}/build/native/include)
    set(fpga_core_lib ${lotus_FPGA_CORE_LIB_PATH}/build/native/lib/x64/dynamic/FPGACoreLib.lib)
    #TODO: this is the pre-build v2 bs_client lib, we might need to build it offline for code-gen case
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set(bs_client_lib ${lotus_BS_CLIENT_PACKAGE}/x64/Debug/client.lib)
    elseif (CMAKE_BUILD_TYPE MATCHES Release)
        set(bs_client_lib ${lotus_BS_CLIENT_PACKAGE}/x64/Release/client.lib)
    endif()
    if (MSVC)
      set(DISABLED_WARNINGS_FOR_FPGA "/wd4996" "/wd4200")
    endif()
  else()
    message(FATAL_ERROR "BrainSlice only works on Windows")
  endif()
  
  file(GLOB_RECURSE onnxruntime_providers_brainslice_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/brainslice/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/brainslice/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_brainslice_cc_srcs})
  add_library(onnxruntime_providers_brainslice ${onnxruntime_providers_brainslice_cc_srcs})
  target_link_libraries(onnxruntime_providers_brainslice ${fpga_core_lib} ${bs_client_lib})
  onnxruntime_add_include_to_target(onnxruntime_providers_brainslice onnx protobuf::libprotobuf)
  target_include_directories(onnxruntime_providers_brainslice PRIVATE ${ONNXRUNTIME_ROOT})
  target_include_directories(onnxruntime_providers_brainslice PRIVATE 
                             ${lotus_BS_CLIENT_PACKAGE}/inc )
  add_definitions(-DBOOST_LOCALE_NO_LIB)
  add_dependencies(onnxruntime_providers_brainslice gsl onnx)
  set_target_properties(onnxruntime_providers_brainslice PROPERTIES FOLDER "Lotus")
  set_target_properties(onnxruntime_providers_brainslice PROPERTIES LINKER_LANGUAGE CXX)
  target_compile_options(onnxruntime_providers_brainslice PRIVATE ${DISABLED_WARNINGS_FOR_FPGA})
endif()

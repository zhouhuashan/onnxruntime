# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# internal providers
# this file is meant to be included in the parent onnxruntime_providers.cmake file

# TODO: move nuphar out of internal
set(onnxruntime_USE_NUPHAR ON)

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

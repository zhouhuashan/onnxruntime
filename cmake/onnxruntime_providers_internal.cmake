# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# internal providers
# this file is meant to be included in the parent onnxruntime_providers.cmake file

# nuphar
if (lotus_USE_TVM)
  file(GLOB_RECURSE onnxruntime_providers_nuphar_cc_srcs
    "${LOTUS_ROOT}/core/providers/nuphar/*.h"
    "${LOTUS_ROOT}/core/providers/nuphar/*.cc"
  )

  source_group(TREE ${LOTUS_ROOT}/core FILES ${onnxruntime_providers_nuphar_cc_srcs})
  add_library(onnxruntime_providers_nuphar ${onnxruntime_providers_nuphar_cc_srcs})
  lotus_add_include_to_target(onnxruntime_providers_nuphar onnx protobuf::libprotobuf)
  set_target_properties(onnxruntime_providers_nuphar PROPERTIES FOLDER "Lotus")
  target_include_directories(onnxruntime_providers_nuphar PRIVATE ${LOTUS_ROOT} ${TVM_INCLUDES})
  set_target_properties(onnxruntime_providers_nuphar PROPERTIES LINKER_LANGUAGE CXX)
  target_compile_options(onnxruntime_providers_nuphar PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
  add_dependencies(onnxruntime_providers_nuphar ${onnxruntime_tvm_dependencies})

  # use this if you want this provider to be included in the onnxruntime shared library
  list(APPEND onnxruntime_libs onnxruntime_providers_nuphar)

endif()

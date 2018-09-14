# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_codegen_srcs  
    "${LOTUS_ROOT}/core/codegen_utils/*.h"
    "${LOTUS_ROOT}/core/codegen_utils/*.cc"
)

add_library(onnxruntime_codegen_utils ${onnxruntime_codegen_srcs})
set_target_properties(onnxruntime_codegen_utils PROPERTIES FOLDER "Lotus")
target_include_directories(onnxruntime_codegen_utils PRIVATE ${LOTUS_ROOT} ${TVM_INCLUDES})
lotus_add_include_to_target(onnxruntime_codegen_utils onnx protobuf::libprotobuf)
target_compile_options(onnxruntime_codegen_utils PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
# need onnx to build to create headers that this project includes
add_dependencies(onnxruntime_codegen_utils onnxruntime_framework tvm ${lotus_EXTERNAL_DEPENDENCIES})


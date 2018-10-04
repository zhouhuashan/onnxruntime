# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_framework_srcs
    "${ONNXRUNTIME_INCLUDE_DIR}/core/framework/*.h"    
    "${ONNXRUNTIME_ROOT}/core/framework/*.h"
    "${ONNXRUNTIME_ROOT}/core/framework/*.cc"
)

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_framework_srcs})

add_library(onnxruntime_framework ${onnxruntime_framework_srcs})

#TODO: remove ${eigen_INCLUDE_DIRS} from here
target_include_directories(onnxruntime_framework PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
onnxruntime_add_include_to_target(onnxruntime_framework onnx protobuf::libprotobuf)
set_target_properties(onnxruntime_framework PROPERTIES FOLDER "Lotus")
# need onnx to build to create headers that this project includes
add_dependencies(onnxruntime_framework ${onnxruntime_EXTERNAL_DEPENDENCIES})

if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include. 
    set_target_properties(onnxruntime_framework PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()

file(GLOB_RECURSE onnxruntime_util_srcs
    "${ONNXRUNTIME_ROOT}/core/util/*.h"
    "${ONNXRUNTIME_ROOT}/core/util/*.cc"
)

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_util_srcs})

add_library(onnxruntime_util ${onnxruntime_util_srcs})
target_include_directories(onnxruntime_util PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
onnxruntime_add_include_to_target(onnxruntime_util onnx protobuf::libprotobuf)
set_target_properties(onnxruntime_util PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_util PROPERTIES FOLDER "Lotus")
add_dependencies(onnxruntime_util ${onnxruntime_EXTERNAL_DEPENDENCIES} eigen)
if (WIN32)
    target_compile_definitions(onnxruntime_util PRIVATE _SCL_SECURE_NO_WARNINGS)
    target_compile_definitions(onnxruntime_framework PRIVATE _SCL_SECURE_NO_WARNINGS)
endif()
if (onnxruntime_USE_MLAS AND WIN32)
  target_include_directories(onnxruntime_util PRIVATE ${ONNXRUNTIME_ROOT} ${MLAS_INC})
endif()


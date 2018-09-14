# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# have to add an empty source file to satisfy cmake when building a shared lib
# from archives only
add_library(onnxruntime SHARED ${LOTUS_ROOT}/core/framework/empty.cc)
set(BEGIN_WHOLE_ARCHIVE -Wl,--whole-archive)
set(END_WHOLE_ARCHIVE -Wl,--no-whole-archive)

set(onnxruntime_libs 
  onnxruntime_session
  ${LOTUS_PROVIDERS_MKLDNN}
  ${LOTUS_PROVIDERS_NUPHAR}
  onnxruntime_providers
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnx
  onnx_proto
  onnxruntime_common
  )

set(onnxruntime_dependencies
  ${lotus_EXTERNAL_DEPENDENCIES}
  )

add_dependencies(onnxruntime ${onnxruntime_dependencies})

target_link_libraries(onnxruntime
  ${BEGIN_WHOLE_ARCHIVE}
  ${onnxruntime_libs}
  ${END_WHOLE_ARCHIVE}
  ${lotus_EXTERNAL_LIBRARIES})

set_target_properties(onnxruntime PROPERTIES LINK_FLAGS "-Wl,-rpath,\$ORIGIN")
set_target_properties(onnxruntime PROPERTIES FOLDER "Lotus")

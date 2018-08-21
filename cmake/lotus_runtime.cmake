add_library(lotus_runtime SHARED)
set(BEGIN_WHOLE_ARCHIVE -Wl,--whole-archive)
set(END_WHOLE_ARCHIVE -Wl,--no-whole-archive)

set(lotus_runtime_libs 
  lotus_session
  ${LOTUS_PROVIDERS_MKLDNN}
  lotus_providers
  lotus_framework
  lotus_util
  lotusIR_graph
  onnx
  onnx_proto
  lotus_common
  )

set(lotus_runtime_dependencies
  ${lotus_EXTERNAL_DEPENDENCIES}
  )

add_dependencies(lotus_runtime ${lotus_runtime_dependencies})

target_link_libraries(lotus_runtime
  ${BEGIN_WHOLE_ARCHIVE}
  ${lotus_runtime_libs}
  ${END_WHOLE_ARCHIVE}
  ${lotus_EXTERNAL_LIBRARIES})

set_target_properties(lotus_runtime PROPERTIES LINK_FLAGS "-Wl,-rpath,\$ORIGIN")
set_target_properties(lotus_runtime PROPERTIES FOLDER "Lotus")

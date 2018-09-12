file(GLOB onnxruntime_session_srcs
    "${LOTUS_INCLUDE_DIR}/core/session/*.h"
    "${LOTUS_ROOT}/core/session/*.h"
    "${LOTUS_ROOT}/core/session/*.cc"
    )

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_session_srcs})

add_library(onnxruntime_session ${onnxruntime_session_srcs})
lotus_add_include_to_target(onnxruntime_session onnx protobuf::libprotobuf)
target_include_directories(onnxruntime_session PRIVATE ${LOTUS_ROOT})
add_dependencies(onnxruntime_session ${lotus_EXTERNAL_DEPENDENCIES})

set_target_properties(onnxruntime_session PROPERTIES FOLDER "Lotus")

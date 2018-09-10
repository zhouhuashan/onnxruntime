file(GLOB lotus_session_srcs
    "${LOTUS_INCLUDE_DIR}/core/session/*.h"
    "${LOTUS_ROOT}/core/session/*.h"
    "${LOTUS_ROOT}/core/session/*.cc"
    )

source_group(TREE ${REPO_ROOT} FILES ${lotus_session_srcs})

add_library(lotus_session ${lotus_session_srcs})
lotus_add_include_to_target(lotus_session onnx protobuf::libprotobuf)
target_include_directories(lotus_session PRIVATE ${LOTUS_ROOT})
add_dependencies(lotus_session ${lotus_EXTERNAL_DEPENDENCIES})

set_target_properties(lotus_session PROPERTIES FOLDER "Lotus")

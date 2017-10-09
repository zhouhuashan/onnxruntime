file(GLOB_RECURSE onnx_protos_src RELATIVE ${CMAKE_CURRENT_BINARY_DIR}/../..
    "${CMAKE_CURRENT_BINARY_DIR}/../../external/onnx/onnx/*.proto"
)

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
     ${CMAKE_CURRENT_BINARY_DIR}/../.. ${onnx_protos_src}
)

add_library(onnx_protos ${PROTO_SRCS} ${PROTO_HDRS})

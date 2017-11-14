RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
    ${LOTUSIR_ROOT} external/onnx/onnx/onnx-ml.proto
)

add_library(onnx_protos ${PROTO_SRCS} ${PROTO_HDRS})
if (WIN32)
    target_compile_options(onnx_protos PRIVATE
        /wd4800 # 'type' : forcing value to bool 'true' or 'false' (performance warning)
        /wd4125 # decimal digit terminates octal escape sequence
    )
endif()

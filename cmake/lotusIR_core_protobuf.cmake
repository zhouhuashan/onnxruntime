file(GLOB_RECURSE lotusIR_protos_src RELATIVE ${LOTUSIR_ROOT}
    "${LOTUSIR_ROOT}/core/protobuf/*.proto"
)

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
    ${LOTUSIR_ROOT} ${lotusIR_protos_src}
)

add_library(lotusIR_protos ${PROTO_SRCS} ${PROTO_HDRS})
if (WIN32)
    target_compile_options(lotusIR_protos PRIVATE
        /wd4800 # 'type' : forcing value to bool 'true' or 'false' (performance warning)
        /wd4125 # decimal digit terminates octal escape sequence
    )
endif()

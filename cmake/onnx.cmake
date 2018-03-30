RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
    ${LOTUS_ROOT} onnx/onnx-ml.proto
)

file(GLOB_RECURSE onnx_src
    "${LOTUS_ROOT}/../external/onnx/onnx/*.h"
    "${LOTUS_ROOT}/../external/onnx/onnx/*.cc"
)

file(GLOB_RECURSE onnx_exclude_src
    "${LOTUS_ROOT}/../external/onnx/onnx/py_utils.h"
    "${LOTUS_ROOT}/../external/onnx/onnx/proto_utils.h"
    "${LOTUS_ROOT}/../external/onnx/onnx/cpp2py_export.cc"
)

list(REMOVE_ITEM onnx_src ${onnx_exclude_src})
add_library(onnx ${PROTO_SRCS} ${PROTO_HDRS} ${onnx_src})

add_dependencies(onnx protobuf)

if (WIN32)
    target_compile_options(onnx PRIVATE
        /wd4800 # 'type' : forcing value to bool 'true' or 'false' (performance warning)
        /wd4125 # decimal digit terminates octal escape sequence
        /wd4100 # 'param' : unreferenced formal parameter
        /wd4244 # 'argument' conversion from 'google::protobuf::int64' to 'int', possible loss of data
        /EHsc   # exception handling - C++ may throw, extern "C" will not
    )
    set(onnx_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(onnx PROPERTIES
        STATIC_LIBRARY_FLAGS "${onnx_static_library_flags}")
endif()
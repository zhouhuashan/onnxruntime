file(GLOB_RECURSE lotusIR_graph_src
    "${LOTUS_ROOT}/core/graph/*.h"
    "${LOTUS_ROOT}/core/graph/*.cc"
)

file(GLOB_RECURSE lotusIR_defs_src
    "${LOTUS_ROOT}/core/defs/*.cc"
)

add_library(lotusIR_graph ${lotusIR_graph_src} ${lotusIR_defs_src})
target_link_libraries(lotusIR_graph PUBLIC onnx)
if (WIN32)
    set(lotusIR_graph_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(lotusIR_graph PROPERTIES
        STATIC_LIBRARY_FLAGS "${lotusIR_graph_static_library_flags}")
    target_compile_options(lotusIR_graph PRIVATE
        /EHsc   # exception handling - C++ may throw, extern "C" will not
    )
endif()

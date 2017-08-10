file(GLOB_RECURSE commonIR_graph_src
    "${COMMONIR_ROOT}/core/graph//*.h"
    "${COMMONIR_ROOT}/core/graph//*.cc"
)

add_library(commonIR_graph ${commonIR_graph_src})
add_dependencies(commonIR_graph commonIR_protos)
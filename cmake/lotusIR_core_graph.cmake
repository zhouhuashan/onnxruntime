file(GLOB_RECURSE lotusIR_graph_src
    "${LOTUSIR_ROOT}/core/graph//*.h"
    "${LOTUSIR_ROOT}/core/graph//*.cc"
)

set(lotusIR_graph_header ${LOTUSIR_ROOT}/core/graph)

add_library(lotusIR_graph ${lotusIR_graph_src})
add_dependencies(lotusIR_graph lotusIR_protos)

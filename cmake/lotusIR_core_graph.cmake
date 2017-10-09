file(GLOB_RECURSE lotusIR_graph_src
    "${LOTUSIR_ROOT}/core/graph//*.h"
    "${LOTUSIR_ROOT}/core/graph//*.cc"
)

file(GLOB_RECURSE lotusIR_defs_src
    "${LOTUSIR_ROOT}/core/defs//*.cc"
)

include_directories(${LOTUSIR_ROOT}/core/graph)

add_library(lotusIR_graph ${lotusIR_graph_src} ${lotusIR_defs_src})
target_link_libraries(lotusIR_graph PUBLIC lotusIR_protos)

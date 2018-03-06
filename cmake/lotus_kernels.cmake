set(lotus_core_operators_src_patterns
    "${LOTUS_ROOT}/core/kernels/*.h"
    "${LOTUS_ROOT}/core/kernels/*.cc"
)

file(GLOB_RECURSE lotus_core_operators_src ${lotus_core_operators_src_patterns})

add_library(lotus_operator_kernels ${lotus_core_operators_src})
source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_core_operators_src})

target_link_libraries(lotus_operator_kernels PUBLIC lotus_framework onnx lotusIR_graph)

SET_TARGET_PROPERTIES(lotus_operator_kernels PROPERTIES LINKER_LANGUAGE CXX)


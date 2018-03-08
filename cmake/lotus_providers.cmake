
file(GLOB_RECURSE lotus_providers_srcs
    "${LOTUS_ROOT}/core/providers/cpu/*.h"
    "${LOTUS_ROOT}/core/providers/cpu/*.cc"
)

add_library(lotus_providers_obj OBJECT ${lotus_providers_srcs})
source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_providers_srcs})

add_dependencies(lotus_providers_obj lotus_core_framework_obj)
SET_TARGET_PROPERTIES(lotus_providers_obj PROPERTIES LINKER_LANGUAGE CXX)

add_library(lotus_providers $<TARGET_OBJECTS:lotus_providers_obj>)
target_link_libraries(lotus_providers PUBLIC lotus_framework PRIVATE ${lotus_EXTERNAL_LIBRARIES})
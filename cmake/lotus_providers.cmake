file(GLOB_RECURSE lotus_providers_srcs
    "${LOTUS_ROOT}/core/providers/cpu/*.h"
    "${LOTUS_ROOT}/core/providers/cpu/*.cc"
)

source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_providers_srcs})

add_library(lotus_providers_obj OBJECT ${lotus_providers_srcs})

add_dependencies(lotus_providers_obj eigen gsl)

set_target_properties(lotus_providers_obj PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(lotus_providers_obj PROPERTIES FOLDER "Lotus")

add_library(lotus_providers $<TARGET_OBJECTS:lotus_providers_obj>)
target_link_libraries(lotus_providers PUBLIC lotus_framework lotus_common PRIVATE ${lotus_EXTERNAL_LIBRARIES})
set_target_properties(lotus_providers PROPERTIES FOLDER "Lotus")

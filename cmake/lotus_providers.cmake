
file(GLOB_RECURSE lotus_provider_srcs
	"${LOTUS_ROOT}/core/providers/cpu/*.h"
    "${LOTUS_ROOT}/core/providers/cpu/*.cc"
)

add_library(lotus_provider_obj OBJECT ${lotus_provider_srcs})
source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_provider_srcs})

add_dependencies(lotus_provider_obj lotus_core_framework_obj)
SET_TARGET_PROPERTIES(lotus_provider_obj PROPERTIES LINKER_LANGUAGE CXX)

add_library(lotus_provider $<TARGET_OBJECTS:lotus_provider_obj>)
target_link_libraries(lotus_provider PUBLIC lotus_framework PRIVATE ${lotus_EXTERNAL_LIBRARIES})
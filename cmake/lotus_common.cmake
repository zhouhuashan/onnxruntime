set(lotus_core_common_src_patterns
    "${LOTUS_ROOT}/core/common/*.h"
    "${LOTUS_ROOT}/core/common/*.cc"
)

if(WIN32)
    list(APPEND lotus_core_common_src_patterns
         "${LOTUS_ROOT}/core/platform/windows/*.h"
         "${LOTUS_ROOT}/core/platform/windows/*.cc"
    )
else()
    list(APPEND lotus_core_common_src_patterns
         "${LOTUS_ROOT}/core/platform/posix/*.h"
         "${LOTUS_ROOT}/core/platform/posix/*.cc"
    )
endif()

file(GLOB lotus_core_common_src ${lotus_core_common_src_patterns})

add_library(lotus_core_common OBJECT ${lotus_core_common_src})
SET_TARGET_PROPERTIES(lotus_core_common PROPERTIES LINKER_LANGUAGE CXX)


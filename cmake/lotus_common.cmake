set(lotus_core_common_src_patterns
    "${LOTUS_ROOT}/core/common/*.h"
    "${LOTUS_ROOT}/core/common/*.cc"
    "${LOTUS_ROOT}/core/lib/*.h"
    "${LOTUS_ROOT}/core/lib/*.cc"
    "${LOTUS_ROOT}/core/platform/env.h"
    "${LOTUS_ROOT}/core/platform/env.cc"
    "${LOTUS_ROOT}/core/platform/env_time.h"
    "${LOTUS_ROOT}/core/platform/env_time.cc"    
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

# artificial hack to build dependencies before runtime if we hit the ALL target
add_dependencies(lotus_core_common ${lotus_EXTERNAL_DEPENDENCIES})

SET_TARGET_PROPERTIES(lotus_core_common PROPERTIES LINKER_LANGUAGE CXX)


set(lotus_common_src_patterns
    "${LOTUS_ROOT}/core/common/*.h"
    "${LOTUS_ROOT}/core/common/*.cc"
    "${LOTUS_ROOT}/core/common/logging/*.h"
    "${LOTUS_ROOT}/core/common/logging/*.cc"
    "${LOTUS_ROOT}/core/common/logging/sinks/*.h"
    "${LOTUS_ROOT}/core/common/logging/sinks/*.cc"
    "${LOTUS_ROOT}/core/inc/*.h"
    "${LOTUS_ROOT}/core/lib/*.h"
    "${LOTUS_ROOT}/core/lib/*.cc"
    "${LOTUS_ROOT}/core/platform/env.h"
    "${LOTUS_ROOT}/core/platform/env.cc"
    "${LOTUS_ROOT}/core/platform/env_time.h"
    "${LOTUS_ROOT}/core/platform/env_time.cc"    
)

if(WIN32)
    list(APPEND lotus_common_src_patterns
         "${LOTUS_ROOT}/core/platform/windows/*.h"
         "${LOTUS_ROOT}/core/platform/windows/*.cc"
         "${LOTUS_ROOT}/core/platform/windows/logging/*.h"
         "${LOTUS_ROOT}/core/platform/windows/logging/*.cc"    
    )
else()
    list(APPEND lotus_common_src_patterns
         "${LOTUS_ROOT}/core/platform/posix/*.h"
         "${LOTUS_ROOT}/core/platform/posix/*.cc"
    )
endif()

file(GLOB lotus_common_src ${lotus_common_src_patterns})

source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_common_src})

add_library(lotus_common ${lotus_common_src})
target_include_directories(lotus_common PRIVATE ${date_INCLUDE_DIR})
# logging uses date. threadpool uses eigen
add_dependencies(lotus_common date eigen gsl)

set_target_properties(lotus_common PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(lotus_common PROPERTIES FOLDER "Lotus")

if(WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include. 
    set_target_properties(lotus_common PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
endif()



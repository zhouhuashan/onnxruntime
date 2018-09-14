# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(onnxruntime_common_src_patterns
    "${LOTUS_INCLUDE_DIR}/core/common/*.h"    
    "${LOTUS_INCLUDE_DIR}/core/common/logging/*.h"    
    "${LOTUS_ROOT}/core/common/*.h"
    "${LOTUS_ROOT}/core/common/*.cc"
    "${LOTUS_ROOT}/core/common/logging/*.h"
    "${LOTUS_ROOT}/core/common/logging/*.cc"
    "${LOTUS_ROOT}/core/common/logging/sinks/*.h"
    "${LOTUS_ROOT}/core/common/logging/sinks/*.cc"
    "${LOTUS_ROOT}/core/inc/*.h"
    "${LOTUS_ROOT}/core/platform/env.h"
    "${LOTUS_ROOT}/core/platform/env.cc"
    "${LOTUS_ROOT}/core/platform/env_time.h"
    "${LOTUS_ROOT}/core/platform/env_time.cc"    
)

if(WIN32)
    list(APPEND onnxruntime_common_src_patterns
         "${LOTUS_ROOT}/core/platform/windows/*.h"
         "${LOTUS_ROOT}/core/platform/windows/*.cc"
         "${LOTUS_ROOT}/core/platform/windows/logging/*.h"
         "${LOTUS_ROOT}/core/platform/windows/logging/*.cc"    
    )
else()
    list(APPEND onnxruntime_common_src_patterns
         "${LOTUS_ROOT}/core/platform/posix/*.h"
         "${LOTUS_ROOT}/core/platform/posix/*.cc"
    )
endif()

file(GLOB onnxruntime_common_src ${onnxruntime_common_src_patterns})

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_common_src})
 
add_library(onnxruntime_common ${onnxruntime_common_src})

if(NOT WIN32)
	target_link_libraries(onnxruntime_common dl)
endif()
target_include_directories(onnxruntime_common PRIVATE ${LOTUS_ROOT} ${date_INCLUDE_DIR})
# logging uses date. threadpool uses eigen
add_dependencies(onnxruntime_common date eigen gsl)

set_target_properties(onnxruntime_common PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_common PROPERTIES FOLDER "Lotus")

if(WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include. 
    set_target_properties(onnxruntime_common PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
endif()



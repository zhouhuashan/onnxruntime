set(lotus_common_src_patterns
    "${LOTUS_ROOT}/core/common/*.h"
    "${LOTUS_ROOT}/core/common/*.cc"
    "${LOTUS_ROOT}/core/common/logging/*.h"
    "${LOTUS_ROOT}/core/common/logging/*.cc"
    "${LOTUS_ROOT}/core/common/logging/sinks/*.h"
    "${LOTUS_ROOT}/core/common/logging/sinks/*.cc"
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

add_library(lotus_common_obj OBJECT ${lotus_common_src})

# logging uses date. threadpool uses eigen
add_dependencies(lotus_common_obj date eigen)

set_target_properties(lotus_common_obj PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(lotus_common_obj PROPERTIES FOLDER "Lotus")

if(WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include. 
    set_target_properties(lotus_common_obj PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()

# add library that tests can link against
add_library(lotus_common 
            $<TARGET_OBJECTS:lotus_common_obj>)            

set_target_properties(lotus_common PROPERTIES FOLDER "Lotus")

if (WIN32)
    target_compile_definitions(lotus_common PRIVATE
        _SCL_SECURE_NO_WARNINGS
    )

    set(lotus_common_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    
    set_target_properties(lotus_common PROPERTIES
        STATIC_LIBRARY_FLAGS "${lotus_common_static_library_flags}"
    )
    
    target_compile_options(lotus_common PRIVATE
        /EHsc   # exception handling - C++ may throw, extern "C" will not
    )
endif()


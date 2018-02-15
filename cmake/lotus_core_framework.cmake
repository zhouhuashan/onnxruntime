file(GLOB_RECURSE lotus_core_framework_srcs
    "${LOTUS_ROOT}/core/framework/*.h"
    "${LOTUS_ROOT}/core/framework/*.cc"
)

set(lotus_core_platform_src_patterns
    "${LOTUS_ROOT}/core/platform/logging.h"
    "${LOTUS_ROOT}/core/platform/log_sink.h"
    "${LOTUS_ROOT}/core/platform/log_sink_common.cc"
    "${LOTUS_ROOT}/core/platform/logging.cc"
)
if(WIN32)
    list(APPEND lotus_core_platform_src_patterns
         "${LOTUS_ROOT}/core/platform/windows/*.h"
         "${LOTUS_ROOT}/core/platform/windows/*.cc"
    )
else()
    list(APPEND lotus_core_platform_src_patterns
         "${LOTUS_ROOT}/core/platform/posix/*.h"
         "${LOTUS_ROOT}/core/platform/posix/*.cc"
    )
endif()

file(GLOB lotus_core_platform_srcs ${lotus_core_platform_src_patterns})

list(APPEND lotus_core_srcs
     ${lotus_core_framework_srcs}
     ${lotus_core_platform_srcs}
)

add_library(lotus_framework ${lotus_core_srcs})
target_link_libraries(lotus_framework PUBLIC onnx lotusIR_graph PRIVATE ${protobuf_STATIC_LIBRARIES})
if (WIN32)
    set(lotus_framework_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(lotus_framework PROPERTIES
        STATIC_LIBRARY_FLAGS "${lotus_framework_static_library_flags}")
    target_compile_options(lotus_framework PRIVATE
        /EHsc   # exception handling - C++ may throw, extern "C" will not
    )
endif()

file(GLOB_RECURSE lotus_core_framework_srcs
    "${LOTUS_ROOT}/core/framework/*.h"
    "${LOTUS_ROOT}/core/framework/*.cc"
)

add_library(lotus_core_framework_obj OBJECT ${lotus_core_framework_srcs})
add_dependencies(lotus_core_framework_obj lotusIR_graph)
if (WIN32)
	set(lotus_framework_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(lotus_core_framework_obj PROPERTIES
        STATIC_LIBRARY_FLAGS "${lotus_framework_static_library_flags}")
    target_compile_options(lotus_core_framework_obj PRIVATE
        /EHsc   # exception handling - C++ may throw, extern "C" will not
    )
	
endif()

file(GLOB_RECURSE lotus_util_srcs
	"${LOTUS_ROOT}/core/util/*.h"
    "${LOTUS_ROOT}/core/util/*.cc"
)

add_library(lotus_util_obj OBJECT ${lotus_util_srcs})
add_dependencies(lotus_util_obj lotus_core_framework_obj)
SET_TARGET_PROPERTIES(lotus_util_obj PROPERTIES LINKER_LANGUAGE CXX)
if (WIN32)
    target_compile_definitions(lotus_util_obj PRIVATE
        _SCL_SECURE_NO_WARNINGS
    )	
endif()

add_library(lotus_framework 
            $<TARGET_OBJECTS:lotus_core_framework_obj>
			$<TARGET_OBJECTS:lotus_util_obj>)
target_link_libraries(lotus_framework PUBLIC onnx lotusIR_graph PRIVATE ${lotus_EXTERNAL_LIBRARIES})
if (WIN32)
    target_compile_definitions(lotus_framework PRIVATE
        _SCL_SECURE_NO_WARNINGS
    )
	
    set(lotus_framework_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(lotus_framework PROPERTIES
        STATIC_LIBRARY_FLAGS "${lotus_framework_static_library_flags}")
    target_compile_options(lotus_framework PRIVATE
        /EHsc   # exception handling - C++ may throw, extern "C" will not
    )
endif()

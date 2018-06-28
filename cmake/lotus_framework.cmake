file(GLOB_RECURSE lotus_framework_srcs
    "${LOTUS_ROOT}/core/framework/*.h"
    "${LOTUS_ROOT}/core/framework/*.cc"
)

if(lotus_USE_CUDA)
    file(GLOB_RECURSE lotus_framework_cuda_srcs     "${LOTUS_ROOT}/core/framework/*.cu")
    list(APPEND lotus_framework_srcs ${lotus_framework_cuda_srcs})
endif()

source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_framework_srcs})

add_library(lotus_framework_obj OBJECT ${lotus_framework_srcs})
set_target_properties(lotus_framework_obj PROPERTIES FOLDER "Lotus")
set_target_properties(lotus_framework_obj PROPERTIES LINKER_LANGUAGE CUDA)
# need onnx to build to create headers that this project includes
add_dependencies(lotus_framework_obj onnx_proto gsl eigen)

if (WIN32)
    set(lotus_framework_static_library_flags
        -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
    )
    set_target_properties(lotus_framework_obj PROPERTIES
        STATIC_LIBRARY_FLAGS "${lotus_framework_static_library_flags}")
    #It's cmake bug, cannot add this compile option for cuda compiler
    #(https://gitlab.kitware.com/cmake/cmake/issues/17535)
    string(APPEND CMAKE_CXX_FLAGS " /EHsc") # exception handling - C++ may throw, extern "C" will not
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include. 
    set_target_properties(lotus_framework_obj PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()

file(GLOB_RECURSE lotus_util_srcs
    "${LOTUS_ROOT}/core/util/*.h"
    "${LOTUS_ROOT}/core/util/*.cc"
)

source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_util_srcs})

add_library(lotus_util_obj OBJECT ${lotus_util_srcs})

set_target_properties(lotus_util_obj PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(lotus_util_obj PROPERTIES FOLDER "Lotus")
add_dependencies(lotus_util_obj onnx gsl ${lotus_EXTERNAL_DEPENDENCIES})

if (WIN32)
    target_compile_definitions(lotus_util_obj PRIVATE
        _SCL_SECURE_NO_WARNINGS
    )   
endif()

add_library(lotus_framework 
            $<TARGET_OBJECTS:lotus_framework_obj>
            $<TARGET_OBJECTS:lotus_util_obj>)
            
target_link_libraries(lotus_framework PUBLIC lotusIR_graph lotus_common onnx PRIVATE ${lotus_EXTERNAL_LIBRARIES})
set_target_properties(lotus_framework PROPERTIES FOLDER "Lotus")

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

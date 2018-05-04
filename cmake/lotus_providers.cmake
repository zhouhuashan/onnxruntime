file(GLOB_RECURSE lotus_providers_srcs
    "${LOTUS_ROOT}/core/providers/cpu/*.h"
    "${LOTUS_ROOT}/core/providers/cpu/*.cc"
)

file(GLOB lotus_providers_common_srcs
    "${LOTUS_ROOT}/core/providers/*.h"
    "${LOTUS_ROOT}/core/providers/*.cc"
    )
    
source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_providers_common_srcs} ${lotus_providers_srcs})

add_library(lotus_providers_obj OBJECT ${lotus_providers_common_srcs} ${lotus_providers_srcs})

add_dependencies(lotus_providers_obj eigen gsl onnx)

set_target_properties(lotus_providers_obj PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(lotus_providers_obj PROPERTIES FOLDER "Lotus")

add_library(lotus_providers $<TARGET_OBJECTS:lotus_providers_obj>)
target_link_libraries(lotus_providers PUBLIC lotus_framework lotus_common PRIVATE ${lotus_EXTERNAL_LIBRARIES})
set_target_properties(lotus_providers PROPERTIES FOLDER "Lotus")
if (lotus_USE_CUDA)
    file(GLOB_RECURSE lotus_providers_cuda_srcs
        "${LOTUS_ROOT}/core/providers/cuda/*.h"
        "${LOTUS_ROOT}/core/providers/cuda/*.cu"
        "${LOTUS_ROOT}/core/providers/cuda/*.cc"
    )
    source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_providers_cuda_srcs})
    add_library(lotus_providers_cuda_obj OBJECT ${lotus_providers_cuda_srcs})
    set_target_properties(lotus_providers_cuda_obj PROPERTIES LINKER_LANGUAGE CUDA)
    set_target_properties(lotus_providers_cuda_obj PROPERTIES FOLDER "Lotus")

    add_library(lotus_providers_cuda $<TARGET_OBJECTS:lotus_providers_cuda_obj>)
    target_link_libraries(lotus_providers_cuda PUBLIC lotus_framework lotus_common PRIVATE ${lotus_EXTERNAL_LIBRARIES})
    set_target_properties(lotus_providers_cuda PROPERTIES FOLDER "Lotus")
    if (WIN32)
        foreach(src_file ${lotus_providers_cuda_srcs})
            set_source_files_properties(${src_file}
                PROPERTIES
                COMPILE_FLAGS "/Yucuda_pch.h /FIcuda_pch.h")
        endforeach()
        set_source_files_properties("${LOTUS_ROOT}/core/providers/cuda/cuda_pch.cc"
            PROPERTIES
            COMPILE_FLAGS "/Yccuda_pch.h"
        )
    endif()
endif()



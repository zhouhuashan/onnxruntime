file(GLOB_RECURSE lotus_providers_srcs
    "${LOTUS_ROOT}/core/providers/cpu/*.h"
    "${LOTUS_ROOT}/core/providers/cpu/*.cc"
)

file(GLOB lotus_providers_common_srcs
    "${LOTUS_ROOT}/core/providers/*.h"
    "${LOTUS_ROOT}/core/providers/*.cc"
    )
    
source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_providers_common_srcs} ${lotus_providers_srcs})
add_library(lotus_providers ${lotus_providers_common_srcs} ${lotus_providers_srcs})
lotus_add_include_to_target(lotus_providers onnx protobuf::libprotobuf)
target_include_directories(lotus_providers PRIVATE ${MLAS_INC} ${eigen_INCLUDE_DIRS})

add_dependencies(lotus_providers eigen gsl onnx)

set_target_properties(lotus_providers PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(lotus_providers PROPERTIES FOLDER "Lotus")

if (lotus_USE_CUDA)
    file(GLOB_RECURSE lotus_providers_cuda_cc_srcs
        "${LOTUS_ROOT}/core/providers/cuda/*.h"
        "${LOTUS_ROOT}/core/providers/cuda/*.cc"
    )
    file(GLOB_RECURSE lotus_providers_cuda_cu_srcs
        "${LOTUS_ROOT}/core/providers/cuda/*.cu"
        "${LOTUS_ROOT}/core/providers/cuda/*.cuh"
    )
    source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_providers_cuda_cc_srcs})
    #TODO: remove lotus_providers_cuda_cc_obj
    add_library(lotus_providers_cuda_cc_obj OBJECT ${lotus_providers_cuda_cc_srcs})
    lotus_add_include_to_target(lotus_providers_cuda_cc_obj onnx protobuf::libprotobuf)
    add_dependencies(lotus_providers_cuda_cc_obj eigen ${lotus_EXTERNAL_DEPENDENCIES})
    set_target_properties(lotus_providers_cuda_cc_obj PROPERTIES FOLDER "Lotus")
    target_include_directories(lotus_providers_cuda_cc_obj PRIVATE ${eigen_INCLUDE_DIRS})

    add_library(lotus_providers_cuda $<TARGET_OBJECTS:lotus_providers_cuda_cc_obj> ${lotus_providers_cuda_cu_srcs})
    lotus_add_include_to_target(lotus_providers_cuda onnx protobuf::libprotobuf)
    set_target_properties(lotus_providers_cuda PROPERTIES LINKER_LANGUAGE CUDA)
    set_target_properties(lotus_providers_cuda PROPERTIES FOLDER "Lotus")
    target_link_libraries(lotus_providers_cuda PRIVATE lotus_framework lotus_common ${lotus_EXTERNAL_LIBRARIES})
    if (WIN32)
        # *.cu cannot use PCH
        foreach(src_file ${lotus_providers_cuda_cc_srcs})
            set_source_files_properties(${src_file}
                PROPERTIES
                COMPILE_FLAGS "/Yucuda_pch.h /FIcuda_pch.h")
        endforeach()
        set_source_files_properties("${LOTUS_ROOT}/core/providers/cuda/cuda_pch.cc"
            PROPERTIES
            COMPILE_FLAGS "/Yccuda_pch.h"
        )
        
        # disable a warning from the CUDA headers about unreferenced local functions
        if (MSVC)
            target_compile_options(lotus_providers_cuda_cc_obj PRIVATE /wd4505) 
        endif()

    endif()
    set(CUDA_INCLUDE "${CUDA_TOOLKIT_ROOT_DIR}/include")
    set(CUDNN_INCLUDE "${lotus_CUDNN_HOME}/include")
    include_directories(${CUDA_INCLUDE} ${CUDNN_INCLUDE})
endif()

if (lotus_USE_MKLDNN)
    file(GLOB_RECURSE lotus_providers_mkldnn_cc_srcs
        "${LOTUS_ROOT}/core/providers/mkldnn/*.h"
        "${LOTUS_ROOT}/core/providers/mkldnn/*.cc"
    )

    source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_providers_mkldnn_cc_srcs})
    add_library(lotus_providers_mkldnn ${lotus_providers_mkldnn_cc_srcs})
    lotus_add_include_to_target(lotus_providers_mkldnn onnx protobuf::libprotobuf)
    add_dependencies(lotus_providers_mkldnn eigen ${lotus_EXTERNAL_DEPENDENCIES})
    set_target_properties(lotus_providers_mkldnn PROPERTIES FOLDER "Lotus")
    target_include_directories(lotus_providers_mkldnn PRIVATE ${eigen_INCLUDE_DIRS})
    set_target_properties(lotus_providers_mkldnn PROPERTIES LINKER_LANGUAGE CXX)
endif()



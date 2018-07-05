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
target_include_directories(lotus_providers_obj PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES> $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>)

add_dependencies(lotus_providers_obj eigen gsl onnx)

set_target_properties(lotus_providers_obj PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(lotus_providers_obj PROPERTIES FOLDER "Lotus")

add_library(lotus_providers $<TARGET_OBJECTS:lotus_providers_obj>)
set_target_properties(lotus_providers PROPERTIES FOLDER "Lotus")

if (lotus_USE_MLAS AND WIN32)
  include_directories(${PROJECT_SOURCE_DIR}/../../mlas/inc)
  target_link_libraries(lotus_providers PUBLIC lotus_framework lotus_common mlas PRIVATE ${lotus_EXTERNAL_LIBRARIES})
else()
  target_link_libraries(lotus_providers PUBLIC lotus_framework lotus_common PRIVATE ${lotus_EXTERNAL_LIBRARIES})
endif()

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
    add_library(lotus_providers_cuda_cc_obj OBJECT ${lotus_providers_cuda_cc_srcs})
    add_dependencies(lotus_providers_cuda_cc_obj eigen gsl onnx)
    set_target_properties(lotus_providers_cuda_cc_obj PROPERTIES FOLDER "Lotus")
    target_include_directories(lotus_providers_cuda_cc_obj PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES> $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>)

    add_library(lotus_providers_cuda $<TARGET_OBJECTS:lotus_providers_cuda_cc_obj> ${lotus_providers_cuda_cu_srcs})
    set_target_properties(lotus_providers_cuda PROPERTIES LINKER_LANGUAGE CUDA)
    set_target_properties(lotus_providers_cuda PROPERTIES FOLDER "Lotus")
    target_link_libraries(lotus_providers_cuda PUBLIC lotus_framework lotus_common PRIVATE ${lotus_EXTERNAL_LIBRARIES})
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
    endif()
    set(CUDA_INCLUDE "${CUDA_TOOLKIT_ROOT_DIR}/include")
    include_directories(${CUDA_INCLUDE})
endif()



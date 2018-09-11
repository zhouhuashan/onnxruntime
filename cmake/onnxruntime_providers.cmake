file(GLOB_RECURSE onnxruntime_providers_srcs
    "${LOTUS_ROOT}/core/providers/cpu/*.h"
    "${LOTUS_ROOT}/core/providers/cpu/*.cc"
)

file(GLOB_RECURSE lotus_contrib_ops_srcs
    "${LOTUS_ROOT}/contrib_ops/*.h"
    "${LOTUS_ROOT}/contrib_ops/*.cc"
    "${LOTUS_ROOT}/contrib_ops/cpu/*.h"
    "${LOTUS_ROOT}/contrib_ops/cpu/*.cc"
)

file(GLOB onnxruntime_providers_common_srcs
    "${LOTUS_ROOT}/core/providers/*.h"
    "${LOTUS_ROOT}/core/providers/*.cc"
    )
    
source_group(TREE ${LOTUS_ROOT}/core FILES ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs})
add_library(onnxruntime_providers ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs} ${lotus_contrib_ops_srcs})
lotus_add_include_to_target(onnxruntime_providers onnx protobuf::libprotobuf)
target_include_directories(onnxruntime_providers PRIVATE ${MLAS_INC} ${LOTUS_ROOT} ${eigen_INCLUDE_DIRS})

add_dependencies(onnxruntime_providers eigen gsl onnx)

set_target_properties(onnxruntime_providers PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_providers PROPERTIES FOLDER "Lotus")

if (WIN32 AND lotus_USE_OPENMP AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
  add_definitions(/openmp)
endif()

if (lotus_USE_CUDA)
    file(GLOB_RECURSE onnxruntime_providers_cuda_cc_srcs
        "${LOTUS_ROOT}/core/providers/cuda/*.h"
        "${LOTUS_ROOT}/core/providers/cuda/*.cc"
    )
    file(GLOB_RECURSE onnxruntime_providers_cuda_cu_srcs
        "${LOTUS_ROOT}/core/providers/cuda/*.cu"
        "${LOTUS_ROOT}/core/providers/cuda/*.cuh"
    )
    source_group(TREE ${LOTUS_ROOT}/core FILES ${onnxruntime_providers_cuda_cc_srcs})
    if(NOT WIN32)
      set(LOTUS_CUDA_CFLAGS " -std=c++14")
    endif()
    cuda_add_library(onnxruntime_providers_cuda ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_cu_srcs} OPTIONS ${LOTUS_CUDA_CFLAGS})
    lotus_add_include_to_target(onnxruntime_providers_cuda onnx protobuf::libprotobuf)
    add_dependencies(onnxruntime_providers_cuda eigen ${lotus_EXTERNAL_DEPENDENCIES})
    target_include_directories(onnxruntime_providers_cuda PRIVATE ${LOTUS_ROOT} ${lotus_CUDNN_HOME}/include ${eigen_INCLUDE_DIRS})
    set_target_properties(onnxruntime_providers_cuda PROPERTIES LINKER_LANGUAGE CUDA)
    set_target_properties(onnxruntime_providers_cuda PROPERTIES FOLDER "Lotus")
    if (WIN32)
        # *.cu cannot use PCH
        foreach(src_file ${onnxruntime_providers_cuda_cc_srcs})
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
            target_compile_options(onnxruntime_providers_cuda PRIVATE /wd4505) 
        endif()

    endif()
endif()

if (lotus_USE_MKLDNN)
    file(GLOB_RECURSE onnxruntime_providers_mkldnn_cc_srcs
        "${LOTUS_ROOT}/core/providers/mkldnn/*.h"
        "${LOTUS_ROOT}/core/providers/mkldnn/*.cc"
    )

    source_group(TREE ${LOTUS_ROOT}/core FILES ${onnxruntime_providers_mkldnn_cc_srcs})
    add_library(onnxruntime_providers_mkldnn ${onnxruntime_providers_mkldnn_cc_srcs})
    lotus_add_include_to_target(onnxruntime_providers_mkldnn onnx protobuf::libprotobuf)
    add_dependencies(onnxruntime_providers_mkldnn eigen ${lotus_EXTERNAL_DEPENDENCIES})
    set_target_properties(onnxruntime_providers_mkldnn PROPERTIES FOLDER "Lotus")
    target_include_directories(onnxruntime_providers_mkldnn PRIVATE ${LOTUS_ROOT} ${eigen_INCLUDE_DIRS})
    set_target_properties(onnxruntime_providers_mkldnn PROPERTIES LINKER_LANGUAGE CXX)
endif()

if (lotus_USE_TVM)
    file(GLOB_RECURSE onnxruntime_providers_nuphar_cc_srcs
        "${LOTUS_ROOT}/core/providers/nuphar/*.h"
        "${LOTUS_ROOT}/core/providers/nuphar/*.cc"
    )

    source_group(TREE ${LOTUS_ROOT}/core FILES ${onnxruntime_providers_nuphar_cc_srcs})
    add_library(onnxruntime_providers_nuphar ${onnxruntime_providers_nuphar_cc_srcs})
    lotus_add_include_to_target(onnxruntime_providers_nuphar onnx protobuf::libprotobuf)
    set_target_properties(onnxruntime_providers_nuphar PROPERTIES FOLDER "Lotus")
    target_include_directories(onnxruntime_providers_nuphar PRIVATE ${LOTUS_ROOT} ${TVM_INCLUDES})
    set_target_properties(onnxruntime_providers_nuphar PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_options(onnxruntime_providers_nuphar PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
endif()

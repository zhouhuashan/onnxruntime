file(GLOB_RECURSE lotus_codegen_srcs  
    "${LOTUS_ROOT}/core/codegen_utils/*.h"
    "${LOTUS_ROOT}/core/codegen_utils/*.cc"
)

add_library(lotus_codegen_utils ${lotus_codegen_srcs})
set_target_properties(lotus_codegen_utils PROPERTIES FOLDER "Lotus")
target_include_directories(lotus_codegen_utils PRIVATE ${TVM_INCLUDES})
lotus_add_include_to_target(lotus_codegen_utils onnx protobuf::libprotobuf)
if (WIN32 AND MSVC)
    set(DISABLED_WARNINGS_FOR_TVM "/wd4100" "/wd4244" "/wd4275" "/wd4251" "/wd4389")
    target_compile_options(lotus_codegen_utils PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
endif()
# need onnx to build to create headers that this project includes
add_dependencies(lotus_codegen_utils lotus_framework tvm ${lotus_EXTERNAL_DEPENDENCIES})


file(GLOB_RECURSE lotus_framework_srcs
    "${LOTUS_INCLUDE_DIR}/core/framework/*.h"    
    "${LOTUS_ROOT}/core/framework/*.h"
    "${LOTUS_ROOT}/core/framework/*.cc"
)

if(lotus_USE_CUDA)
    file(GLOB_RECURSE lotus_framework_cuda_srcs "${LOTUS_ROOT}/core/framework/*.cu")
    list(APPEND lotus_framework_srcs ${lotus_framework_cuda_srcs})
endif()

source_group(TREE ${REPO_ROOT} FILES ${lotus_framework_srcs})

add_library(lotus_framework ${lotus_framework_srcs})

#TODO: remove ${eigen_INCLUDE_DIRS} from here
target_include_directories(lotus_framework PRIVATE ${LOTUS_ROOT} ${eigen_INCLUDE_DIRS})
lotus_add_include_to_target(lotus_framework onnx protobuf::libprotobuf)
set_target_properties(lotus_framework PROPERTIES FOLDER "Lotus")
if(lotus_USE_CUDA)
  set_target_properties(lotus_framework PROPERTIES LINKER_LANGUAGE CUDA)
endif()
# need onnx to build to create headers that this project includes
add_dependencies(lotus_framework ${lotus_EXTERNAL_DEPENDENCIES} eigen)

if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include. 
    set_target_properties(lotus_framework PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()

file(GLOB_RECURSE lotus_util_srcs
    "${LOTUS_ROOT}/core/util/*.h"
    "${LOTUS_ROOT}/core/util/*.cc"
)

source_group(TREE ${LOTUS_ROOT}/core FILES ${lotus_util_srcs})

add_library(lotus_util ${lotus_util_srcs})
target_include_directories(lotus_util PRIVATE ${LOTUS_ROOT} ${eigen_INCLUDE_DIRS})
lotus_add_include_to_target(lotus_util onnx protobuf::libprotobuf)
set_target_properties(lotus_util PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(lotus_util PROPERTIES FOLDER "Lotus")
add_dependencies(lotus_util ${lotus_EXTERNAL_DEPENDENCIES} eigen)
if (WIN32)
    target_compile_definitions(lotus_util PRIVATE _SCL_SECURE_NO_WARNINGS)
    target_compile_definitions(lotus_framework PRIVATE _SCL_SECURE_NO_WARNINGS)
endif()
if (lotus_USE_MLAS AND WIN32)
  target_include_directories(lotus_util PRIVATE ${LOTUS_ROOT} ${MLAS_INC})
endif()


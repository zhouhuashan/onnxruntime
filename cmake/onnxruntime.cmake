# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_library(onnxruntime SHARED ${onnxruntime_session_srcs})
target_include_directories(onnxruntime PRIVATE ${ONNXRUNTIME_ROOT} ${date_INCLUDE_DIR})
add_dependencies(onnxruntime ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(UNIX)
  set(BEGIN_WHOLE_ARCHIVE -Xlinker --whole-archive)
  set(END_WHOLE_ARCHIVE -Xlinker --no-whole-archive)
  set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/version_script.lds -Xlinker --no-undefined")
else()
  #TODO:WHOLE_ARCHIVE for windows
  target_compile_definitions(onnxruntime PRIVATE ONNX_RUNTIME_EXPORTS=1 ONNX_RUNTIME_BUILD_DLL=1)
endif()

target_link_libraries(onnxruntime PRIVATE
    ${BEGIN_WHOLE_ARCHIVE}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MKLDNN}
    onnxruntime_providers
    onnxruntime_util
    onnxruntime_framework
    ${END_WHOLE_ARCHIVE}
    onnxruntime_graph
    onnxruntime_common
    onnx
    onnx_proto
    onnxruntime_mlas
    ${onnxruntime_EXTERNAL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${ONNXRUNTIME_CUDA_LIBRARIES}
    ${ONNXRUNTIME_SO_LINK_FLAG})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS onnxruntime
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(onnxruntime PROPERTIES FOLDER "Lotus")

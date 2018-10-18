# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# have to add an empty source file to satisfy cmake when building a shared lib
# from archives only

add_library(onnxruntime SHARED ${onnxruntime_session_srcs})
target_include_directories(onnxruntime PRIVATE ${ONNXRUNTIME_ROOT} ${date_INCLUDE_DIR})
add_dependencies(onnxruntime ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(UNIX)
  set(BEGIN_WHOLE_ARCHIVE -Wl,--whole-archive)
  set(END_WHOLE_ARCHIVE -Wl,--no-whole-archive)
  target_link_libraries(onnxruntime PRIVATE
    ${BEGIN_WHOLE_ARCHIVE}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MKLDNN}
    onnxruntime_providers
    onnxruntime_util
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_common
    ${END_WHOLE_ARCHIVE}
    "-Wl,--version-script=${ONNXRUNTIME_ROOT}/core/version_script.lds"
    onnx
    onnx_proto
    onnxruntime_mlas
    ${onnxruntime_EXTERNAL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT} -Wl,--no-undefined)
else()
  target_compile_definitions(onnxruntime PRIVATE ONNX_RUNTIME_EXPORTS=1 ONNX_RUNTIME_BUILD_DLL=1)
  target_link_libraries(onnxruntime PRIVATE
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MKLDNN}
    onnxruntime_providers
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_graph
    onnx
    onnx_proto
    onnxruntime_common
    onnxruntime_mlas
    ${onnxruntime_EXTERNAL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${ONNXRUNTIME_CUDA_LIBRARIES})
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS onnxruntime
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(onnxruntime PROPERTIES FOLDER "Lotus")

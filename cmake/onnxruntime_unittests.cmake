# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(Threads)


set(TEST_SRC_DIR ${ONNXRUNTIME_ROOT}/test)
function(AddTest)
  cmake_parse_arguments(_UT "" "TARGET" "LIBS;SOURCES;DEPENDS" ${ARGN})

  list(REMOVE_DUPLICATES _UT_LIBS)
  list(REMOVE_DUPLICATES _UT_SOURCES)

  if (_UT_DEPENDS)
    list(REMOVE_DUPLICATES _UT_DEPENDS)
  endif(_UT_DEPENDS)

  add_executable(${_UT_TARGET} ${_UT_SOURCES})
  source_group(TREE ${TEST_SRC_DIR} FILES ${_UT_SOURCES})
  set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "LotusTest")

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS} eigen)
  endif(_UT_DEPENDS)

  target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} ${onnxruntime_EXTERNAL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  if (onnxruntime_USE_TVM)
    target_include_directories(${_UT_TARGET} PRIVATE ${MLAS_INC} ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${date_INCLUDE_DIR} ${onnxruntime_CUDNN_HOME}/include ${TVM_INCLUDES})
  else()
    target_include_directories(${_UT_TARGET} PRIVATE ${MLAS_INC} ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${date_INCLUDE_DIR} ${onnxruntime_CUDNN_HOME}/include)
  endif()

  if (WIN32)
    #It's cmake bug, cannot add this compile option for cuda compiler
    #(https://gitlab.kitware.com/cmake/cmake/issues/17535)
    string(APPEND CMAKE_CXX_FLAGS " /EHsc") # exception handling - C++ may throw, extern "C" will not

    if (onnxruntime_USE_CUDA)
      # disable a warning from the CUDA headers about unreferenced local functions
      if (MSVC)
        target_compile_options(${_UT_TARGET} PRIVATE /wd4505)
      endif()
    endif()
    if (onnxruntime_USE_TVM)
      target_compile_options(${_UT_TARGET} PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
    endif()
  endif()

  set(TEST_ARGS)
  if (onnxruntime_GENERATE_TEST_REPORTS)
    # generate a report file next to the test program
    list(APPEND TEST_ARGS
      "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>")
  endif(onnxruntime_GENERATE_TEST_REPORTS)

  add_test(NAME ${_UT_TARGET}
    COMMAND ${_UT_TARGET} ${TEST_ARGS}
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
    )
endfunction(AddTest)

#Do not add '${TEST_SRC_DIR}/util/include' to your include directories directly
#Use onnxruntime_add_include_to_target or target_link_libraries, so that compile definitions
#can propagate correctly.

file(GLOB onnxruntime_test_utils_src
  "${TEST_SRC_DIR}/util/include/*.h"
  "${TEST_SRC_DIR}/util/*.cc"
)

file(GLOB onnxruntime_test_common_src
  "${TEST_SRC_DIR}/common/*.cc"
  "${TEST_SRC_DIR}/common/*.h"
  "${TEST_SRC_DIR}/common/logging/*.cc"
  "${TEST_SRC_DIR}/common/logging/*.h"
  )

file(GLOB onnxruntime_test_ir_src
  "${TEST_SRC_DIR}/ir/*.cc"
  "${TEST_SRC_DIR}/ir/*.h"
  )

set(onnxruntime_test_framework_src_patterns
  "${TEST_SRC_DIR}/framework/*.cc"
  "${TEST_SRC_DIR}/platform/*.cc"
  )

if(WIN32)
  list(APPEND onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/windows/*.cc"
    "${TEST_SRC_DIR}/platform/windows/logging/*.cc" )
endif()

if(onnxruntime_USE_CUDA)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/framework/cuda/*)
endif()

file(GLOB onnxruntime_test_framework_src ${onnxruntime_test_framework_src_patterns})


file(GLOB_RECURSE onnxruntime_test_providers_src
  "${TEST_SRC_DIR}/providers/*.h"
  "${TEST_SRC_DIR}/providers/*.cc"
  "${TEST_SRC_DIR}/contrib_ops/*.h"
  "${TEST_SRC_DIR}/contrib_ops/*.cc"
  ${TEST_SRC_DIR}/framework/TestAllocatorManager.cc
  ${TEST_SRC_DIR}/framework/TestAllocatorManager.h
  )

# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(onnxruntime_test_common_libs
  onnxruntime_test_utils
  onnxruntime_common
  gtest
  gmock
  )

set(onnxruntime_test_ir_libs
  onnxruntime_test_utils
  onnxruntime_graph
  onnx
  onnx_proto
  onnxruntime_common
  protobuf::libprotobuf
  gtest gmock
  )

set(onnxruntime_test_framework_libs
  onnxruntime_test_utils_for_framework
  onnxruntime_session
  onnxruntime_providers
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnx
  onnx_proto
  onnxruntime_common
  protobuf::libprotobuf
  gtest gmock
  )

if(onnxruntime_USE_CUDA)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_cuda)
endif()

if(onnxruntime_USE_MKLDNN)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_mkldnn)
endif()

if(WIN32)
    list(APPEND onnxruntime_test_framework_libs Advapi32)
else()
    list(APPEND onnxruntime_test_framework_libs stdc++fs)
endif()

set(onnxruntime_test_providers_libs
  onnxruntime_test_utils_for_framework
  onnxruntime_session
  ${PROVIDERS_CUDA}
  ${PROVIDERS_MKLDNN}
  onnxruntime_providers
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnx
  onnx_proto
  onnxruntime_common
  protobuf::libprotobuf
  gtest gmock
  )

set (onnxruntime_test_providers_dependencies ${onnxruntime_EXTERNAL_DEPENDENCIES})

if (onnxruntime_USE_MLAS AND WIN32)
  list(APPEND onnxruntime_test_providers_libs ${MLAS_LIBRARY})
endif()

if(onnxruntime_USE_CUDA)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cuda)
endif()

if(onnxruntime_USE_MKLDNN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_mkldnn)
endif()

if( NOT WIN32)
    list(APPEND onnxruntime_test_providers_libs stdc++fs)
endif()

file(GLOB_RECURSE onnxruntime_test_tvm_src
  "${ONNXRUNTIME_ROOT}/test/tvm/*.h"
  "${ONNXRUNTIME_ROOT}/test/tvm/*.cc"
  )

if (onnxruntime_ENABLE_MICROSOFT_INTERNAL)
  include(onnxruntime_unittests_internal.cmake)
endif()

add_library(onnxruntime_test_utils_for_framework ${onnxruntime_test_utils_src})
onnxruntime_add_include_to_target(onnxruntime_test_utils_for_framework gtest onnx protobuf::libprotobuf)
add_dependencies(onnxruntime_test_utils_for_framework ${onnxruntime_EXTERNAL_DEPENDENCIES} eigen)
target_include_directories(onnxruntime_test_utils_for_framework PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE ${eigen_INCLUDE_DIRS} ${ONNXRUNTIME_ROOT})
# Add the define for conditionally using the framework Environment class in TestEnvironment
target_compile_definitions(onnxruntime_test_utils_for_framework PUBLIC -DHAVE_FRAMEWORK_LIB)

if (SingleUnitTestProject)
  add_library(onnxruntime_test_utils ALIAS onnxruntime_test_utils_for_framework)
else()
  add_library(onnxruntime_test_utils ${onnxruntime_test_utils_src})
  onnxruntime_add_include_to_target(onnxruntime_test_utils gtest onnx protobuf::libprotobuf)
  add_dependencies(onnxruntime_test_utils ${onnxruntime_EXTERNAL_DEPENDENCIES} eigen)
  target_include_directories(onnxruntime_test_utils PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE ${eigen_INCLUDE_DIRS})
endif()


if (SingleUnitTestProject)
  set(all_tests ${onnxruntime_test_common_src} ${onnxruntime_test_ir_src} ${onnxruntime_test_framework_src} ${onnxruntime_test_providers_src})
  set(all_libs onnxruntime_test_utils ${onnxruntime_test_providers_libs})
  set(all_dependencies ${onnxruntime_test_providers_dependencies} )

  if (onnxruntime_USE_TVM)
    list(APPEND all_tests ${onnxruntime_test_tvm_src})
    list(APPEND all_libs ${onnxruntime_tvm_libs})
    list(APPEND all_dependencies ${onnxruntime_tvm_dependencies})
  endif()
  # we can only have one 'main', so remove them all and add back the providers test_main as it sets
  # up everything we need for all tests
  file(GLOB_RECURSE test_mains "${TEST_SRC_DIR}/*/test_main.cc")
  list(REMOVE_ITEM all_tests ${test_mains})
  list(APPEND all_tests "${TEST_SRC_DIR}/providers/test_main.cc")

  # this is only added to onnxruntime_test_framework_libs above, but we use onnxruntime_test_providers_libs for the onnxruntime_test_all target.
  # for now, add it here. better is probably to have onnxruntime_test_providers_libs use the full onnxruntime_test_framework_libs
  # list given it's built on top of that library and needs all the same dependencies.
  if(WIN32)
    list(APPEND onnxruntime_test_providers_libs Advapi32)
  endif()

  AddTest(
    TARGET onnxruntime_test_all
    SOURCES ${all_tests}
    LIBS ${all_libs} ${onnxruntime_test_common_libs}
    DEPENDS ${all_dependencies}
  )

  # the default logger tests conflict with the need to have an overall default logger
  # so skip in this type of
  target_compile_definitions(onnxruntime_test_all PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)

  set(test_data_target onnxruntime_test_all)
else()
  AddTest(
    TARGET onnxruntime_test_common
    SOURCES ${onnxruntime_test_common_src}
    LIBS ${onnxruntime_test_common_libs}
    DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
  )

  AddTest(
    TARGET onnxruntime_test_ir
    SOURCES ${onnxruntime_test_ir_src}
    LIBS ${onnxruntime_test_ir_libs}
    DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
  )

  AddTest(
    TARGET onnxruntime_test_framework
    SOURCES ${onnxruntime_test_framework_src}
    LIBS ${onnxruntime_test_framework_libs}
    # code smell! see if CPUExecutionProvider should move to framework so onnxruntime_providers isn't needed.
    DEPENDS ${onnxruntime_test_providers_dependencies}
  )

  AddTest(
    TARGET onnxruntime_test_providers
    SOURCES ${onnxruntime_test_providers_src}
    LIBS ${onnxruntime_test_providers_libs}
    DEPENDS ${onnxruntime_test_providers_dependencies}
  )

  set(test_data_target onnxruntime_test_ir)
endif()  # SingleUnitTestProject

#
# LotusIR_graph test data
#
set(TEST_DATA_SRC ${TEST_SRC_DIR}/testdata)
set(TEST_DATA_DES $<TARGET_FILE_DIR:${test_data_target}>/testdata)

# Copy test data from source to destination.
add_custom_command(
  TARGET ${test_data_target} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${TEST_DATA_SRC}
  ${TEST_DATA_DES})

set(onnx_test_runner_src_dir ${TEST_SRC_DIR}/onnx)
set(onnx_test_runner_common_srcs
  ${onnx_test_runner_src_dir}/TestResultStat.cc
  ${onnx_test_runner_src_dir}/TestResultStat.h
  ${onnx_test_runner_src_dir}/testenv.h
  ${onnx_test_runner_src_dir}/FixedCountFinishCallback.h
  ${onnx_test_runner_src_dir}/TestCaseResult.cc
  ${onnx_test_runner_src_dir}/TestCaseResult.h
  ${onnx_test_runner_src_dir}/testenv.cc
  ${onnx_test_runner_src_dir}/runner.h
  ${onnx_test_runner_src_dir}/runner.cc
  ${onnx_test_runner_src_dir}/TestCase.cc
  ${onnx_test_runner_src_dir}/TestCase.h
  ${onnx_test_runner_src_dir}/sync_api.h
  ${TEST_SRC_DIR}/proto/tml.proto
  )

if(NOT WIN32)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/tml.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
endif()

if(WIN32)
  set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs} ${onnx_test_runner_src_dir}/sync_api_win.cc)
  add_library(win_getopt ${onnx_test_runner_src_dir}/getopt.cc ${onnx_test_runner_src_dir}/getopt.h)
  set_target_properties(win_getopt PROPERTIES FOLDER "LotusTest")
  set(GETOPT_LIB win_getopt)
else()
  set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs} ${onnx_test_runner_src_dir}/onnxruntime_event.h ${onnx_test_runner_src_dir}/simple_thread_pool.h ${onnx_test_runner_src_dir}/sync_api_linux.cc)
  set(FS_STDLIB stdc++fs)
endif()

add_library(onnx_test_runner_common ${onnx_test_runner_common_srcs})
onnxruntime_add_include_to_target(onnx_test_runner_common onnxruntime_test_utils onnx protobuf::libprotobuf)
add_dependencies(onnx_test_runner_common eigen ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnx_test_runner_common PRIVATE ${eigen_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx ${ONNXRUNTIME_ROOT})
set_target_properties(onnx_test_runner_common PROPERTIES FOLDER "LotusTest")

onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS ${ONNXRUNTIME_ROOT}/core/protobuf TARGET onnx_test_runner_common)

if(onnxruntime_USE_CUDA)
  set(onnx_cuda_test_libs onnxruntime_providers_cuda)
endif()

if(onnxruntime_USE_MKLDNN)
  set(onnx_mkldnn_test_libs onnxruntime_providers_mkldnn)
endif()

list(APPEND onnx_test_libs
  onnxruntime_test_utils
  onnxruntime_session
  ${onnx_cuda_test_libs}
  ${onnxruntime_tvm_libs}
  ${onnx_mkldnn_test_libs}
  onnxruntime_providers
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnx
  onnx_proto
  onnxruntime_common
  ${MLAS_LIBRARY}
  ${FS_STDLIB}
  ${onnxruntime_EXTERNAL_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

if (onnxruntime_USE_OPENBLAS)
  if (WIN32)
    list(APPEND onnx_test_libs ${onnxruntime_OPENBLAS_HOME}/lib/libopenblas.lib)
  else()
    list(APPEND onnx_test_libs openblas)
  endif()
endif()

if (onnxruntime_USE_MKLDNN)
  list(APPEND onnx_test_libs mkldnn)
  add_custom_command(
    TARGET ${test_data_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_LIB_DIR}/${MKLDNN_SHARED_LIB} $<TARGET_FILE_DIR:${test_data_target}>
    )
endif()

if (onnxruntime_USE_MKLML)
  add_custom_command(
    TARGET ${test_data_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${MKLDNN_LIB_DIR}/${MKLML_SHARED_LIB} ${MKLDNN_LIB_DIR}/${IOMP5MD_SHARED_LIB}
    $<TARGET_FILE_DIR:${test_data_target}>
    )
endif()

add_executable(onnx_test_runner ${onnx_test_runner_src_dir}/main.cc)
target_link_libraries(onnx_test_runner PRIVATE onnx_test_runner_common ${onnx_test_libs} ${GETOPT_LIB})
target_include_directories(onnx_test_runner PRIVATE ${ONNXRUNTIME_ROOT})
set_target_properties(onnx_test_runner PROPERTIES FOLDER "LotusTest")

if(onnxruntime_BUILD_BENCHMARKS)
  add_executable(onnxruntime_benchmark ${TEST_SRC_DIR}/onnx/microbenchmark/main.cc ${TEST_SRC_DIR}/onnx/microbenchmark/modeltest.cc)
  target_include_directories(onnxruntime_benchmark PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_graph_header} benchmark)
  target_compile_options(onnxruntime_benchmark PRIVATE "/wd4141")
  if (onnxruntime_USE_MLAS AND WIN32)
    target_include_directories(onnxruntime_benchmark PRIVATE ${MLAS_INC})
  endif()
  target_link_libraries(onnxruntime_benchmark PRIVATE ${onnx_test_libs} onnx_test_runner_common benchmark)
  add_dependencies(onnxruntime_benchmark ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_benchmark PROPERTIES FOLDER "LotusTest")
endif()

if(WIN32)
  set(DISABLED_WARNINGS_FOR_PROTOBUF "/wd4125" "/wd4456" "/wd4505")
  target_compile_options(onnx_test_runner_common PRIVATE ${DISABLED_WARNINGS_FOR_PROTOBUF} -D_CRT_SECURE_NO_WARNINGS)
  target_compile_options(onnx_test_runner PRIVATE ${DISABLED_WARNINGS_FOR_PROTOBUF})
  #Maybe "CMAKE_SYSTEM_PROCESSOR" is better
  if(NOT ${CMAKE_GENERATOR_PLATFORM} MATCHES "ARM")
    add_library(onnx_test_runner_vstest SHARED ${onnx_test_runner_src_dir}/vstest_logger.cc ${onnx_test_runner_src_dir}/vstest_main.cc)
    target_compile_options(onnx_test_runner_vstest PRIVATE ${DISABLED_WARNINGS_FOR_PROTOBUF})
    target_include_directories(onnx_test_runner_vstest PRIVATE ${ONNXRUNTIME_ROOT} ${date_INCLUDE_DIR})
    target_link_libraries(onnx_test_runner_vstest PRIVATE ${onnx_test_libs} onnx_test_runner_common)
    set_target_properties(onnx_test_runner_vstest PROPERTIES FOLDER "LotusTest")
  endif()
endif()

set(onnxruntime_exec_src_dir ${TEST_SRC_DIR}/onnxruntime_exec)
file(GLOB onnxruntime_exec_src
  "${onnxruntime_exec_src_dir}/*.cc"
  "${onnxruntime_exec_src_dir}/*.h"
  )
add_executable(onnxruntime_exec ${onnxruntime_exec_src})
target_include_directories(onnxruntime_exec PRIVATE ${ONNXRUNTIME_ROOT})
# we need to force these dependencies to build first. just using target_link_libraries isn't sufficient
add_dependencies(onnxruntime_exec ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_link_libraries(onnxruntime_exec PRIVATE ${onnx_test_libs})
set_target_properties(onnxruntime_exec PROPERTIES FOLDER "LotusTest")

add_test(NAME onnx_test_pytorch_converted
  COMMAND onnx_test_runner ${PROJECT_SOURCE_DIR}/external/onnx/onnx/backend/test/data/pytorch-converted)
add_test(NAME onnx_test_pytorch_operator
  COMMAND onnx_test_runner ${PROJECT_SOURCE_DIR}/external/onnx/onnx/backend/test/data/pytorch-operator)

set(onnxruntime_perf_test_src_dir ${TEST_SRC_DIR}/perftest)
set(onnxruntime_perf_test_src_patterns
  "${onnxruntime_perf_test_src_dir}/*.cc"
  "${onnxruntime_perf_test_src_dir}/*.h")

if(WIN32)
  list(APPEND onnxruntime_perf_test_src_patterns
    "${onnxruntime_perf_test_src_dir}/windows/*.cc"
    "${onnxruntime_perf_test_src_dir}/windows/*.h" )
else ()
  list(APPEND onnxruntime_perf_test_src_patterns
    "${onnxruntime_perf_test_src_dir}/posix/*.cc"
    "${onnxruntime_perf_test_src_dir}/posix/*.h" )
endif()

file(GLOB onnxruntime_perf_test_src ${onnxruntime_perf_test_src_patterns})

add_executable(onnxruntime_perf_test ${onnxruntime_perf_test_src})
target_include_directories(onnxruntime_perf_test PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_graph_header} ${onnx_test_runner_src_dir} ${onnxruntime_exec_src_dir})

target_link_libraries(onnxruntime_perf_test PRIVATE onnx_test_runner_common ${onnx_test_libs} ${GETOPT_LIB})
set_target_properties(onnxruntime_perf_test PROPERTIES FOLDER "LotusTest")

# shared lib
if (onnxruntime_BUILD_SHARED_LIB)
if (UNIX)
  # test custom op shared lib
  file(GLOB onnxruntime_custom_op_shared_lib_test_srcs "${ONNXRUNTIME_ROOT}/test/custom_op_shared_lib/test_custom_op.cc")

  add_library(onnxruntime_custom_op_shared_lib_test SHARED ${onnxruntime_custom_op_shared_lib_test_srcs})
  target_include_directories(onnxruntime_custom_op_shared_lib_test PUBLIC "${PROJECT_SOURCE_DIR}/include")

  target_link_libraries(onnxruntime_custom_op_shared_lib_test
    onnxruntime
    )
  set_target_properties(onnxruntime_custom_op_shared_lib_test PROPERTIES FOLDER "LotusSharedLibTest")

  #################################################################
  # test inference using shared lib + custom op
  file(GLOB onnxruntime_shared_lib_test_srcs "${ONNXRUNTIME_ROOT}/test/shared_lib/test_inference.cc")

  add_executable(onnxruntime_shared_lib_test ${onnxruntime_shared_lib_test_srcs})
  target_include_directories(onnxruntime_shared_lib_test PRIVATE "${PROJECT_SOURCE_DIR}/include" ${ONNXRUNTIME_ROOT})

  target_link_libraries(onnxruntime_shared_lib_test
    onnxruntime
    onnx
    onnx_proto
    )
  set_target_properties(onnxruntime_shared_lib_test PROPERTIES LINK_FLAGS "-Wl,-rpath,\$ORIGIN")
  set_target_properties(onnxruntime_shared_lib_test PROPERTIES FOLDER "LotusSharedLibTest")
endif()
endif()

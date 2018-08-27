find_package(Threads)


set(TEST_SRC_DIR ${LOTUS_ROOT}/test)
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
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif(_UT_DEPENDS)

  target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${lotus_EXTERNAL_LIBRARIES})
  if (lotus_USE_TVM)
    target_include_directories(${_UT_TARGET} PRIVATE ${MLAS_INC} ${eigen_INCLUDE_DIRS} ${date_INCLUDE_DIR} ${TVM_INCLUDESS})
  else(lotus_USE_TVM)
    target_include_directories(${_UT_TARGET} PRIVATE ${MLAS_INC} ${eigen_INCLUDE_DIRS} ${date_INCLUDE_DIR})
  endif()

  if (WIN32)
    #It's cmake bug, cannot add this compile option for cuda compiler
    #(https://gitlab.kitware.com/cmake/cmake/issues/17535)
    string(APPEND CMAKE_CXX_FLAGS " /EHsc") # exception handling - C++ may throw, extern "C" will not

    if (lotus_USE_CUDA)
      # disable a warning from the CUDA headers about unreferenced local functions
      if (MSVC)
        target_compile_options(${_UT_TARGET} PRIVATE /wd4505)
      endif()
    endif()
    if (lotus_USE_TVM)
      target_compile_options(${_UT_TARGET} PRIVATE /wd4100 /wd4244 /wd4275 /wd4251 /wd4389)
    endif()
  endif()

  set(TEST_ARGS)
  if (lotus_GENERATE_TEST_REPORTS)
    # generate a report file next to the test program
    list(APPEND TEST_ARGS
      "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>")
  endif(lotus_GENERATE_TEST_REPORTS)

  add_test(NAME ${_UT_TARGET}
    COMMAND ${_UT_TARGET} ${TEST_ARGS}
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
    )
endfunction(AddTest)

#Do not add '${TEST_SRC_DIR}/util/include' to your include directories directly
#Use lotus_add_include_to_target or target_link_libraries, so that compile definitions
#can propagate correctly.

file(GLOB lotus_test_utils_src
  "${TEST_SRC_DIR}/util/include/*.h"
  "${TEST_SRC_DIR}/util/*.cc"
)

file(GLOB lotus_test_common_src
  "${TEST_SRC_DIR}/common/*.cc"
  "${TEST_SRC_DIR}/common/*.h"
  "${TEST_SRC_DIR}/common/logging/*.cc"
  "${TEST_SRC_DIR}/common/logging/*.h"
  )

file(GLOB lotus_test_ir_src
  "${TEST_SRC_DIR}/ir/*.cc"
  "${TEST_SRC_DIR}/ir/*.h"
  )

set(lotus_test_framework_src_patterns
  "${TEST_SRC_DIR}/framework/*.cc"
  "${TEST_SRC_DIR}/platform/*.cc"
  )

if(WIN32)
  list(APPEND lotus_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/windows/*.cc"
    "${TEST_SRC_DIR}/platform/windows/logging/*.cc" )
endif()

if(lotus_USE_CUDA)
  list(APPEND lotus_test_framework_src_patterns  ${TEST_SRC_DIR}/framework/cuda/*)
  list(APPEND lotus_test_framework_libs ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
endif()

file(GLOB lotus_test_framework_src ${lotus_test_framework_src_patterns})


file(GLOB_RECURSE lotus_test_providers_src
  "${TEST_SRC_DIR}/providers/*.h"
  "${TEST_SRC_DIR}/providers/*.cc"
  ${TEST_SRC_DIR}/framework/TestAllocatorManager.cc
  ${TEST_SRC_DIR}/framework/TestAllocatorManager.h
  )

# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(lotus_test_common_libs
  lotus_test_utils
  lotus_common
  gtest
  gmock
  )

set(lotus_test_ir_libs
  lotus_test_utils
  lotusIR_graph
  onnx
  onnx_proto
  lotus_common
  protobuf::libprotobuf
  gtest gmock
  )

set(lotus_test_framework_libs
  lotus_test_utils_for_framework
  lotus_session
  lotus_providers
  lotus_framework
  lotus_util
  lotusIR_graph
  onnx
  onnx_proto
  lotus_common
  protobuf::libprotobuf
  gtest gmock
  )

if(lotus_USE_CUDA)
  list(APPEND lotus_test_framework_libs lotus_providers_cuda)
endif()

if(lotus_USE_MKLDNN)
  list(APPEND lotus_test_framework_libs lotus_providers_mkldnn)
endif()

if(WIN32)
    list(APPEND lotus_test_framework_libs Advapi32)
else()
    list(APPEND lotus_test_framework_libs stdc++fs)
endif()

set(lotus_test_providers_libs
  lotus_test_utils_for_framework
  lotus_session
  ${LOTUS_PROVIDERS_CUDA}
  ${LOTUS_PROVIDERS_MKLDNN}
  lotus_providers
  lotus_framework
  lotus_util
  lotusIR_graph
  onnx
  onnx_proto
  lotus_common
  protobuf::libprotobuf
  gtest gmock
  )


set (lotus_test_providers_dependencies ${lotus_EXTERNAL_DEPENDENCIES})

if (lotus_USE_MLAS AND WIN32)
  list(APPEND lotus_test_providers_libs ${MLAS_LIBRARY})
endif()

if(lotus_USE_CUDA)
  list(APPEND lotus_test_providers_dependencies lotus_providers_cuda)
  list(APPEND lotus_test_providers_libs ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
endif()

if(lotus_USE_MKLDNN)
  list(APPEND lotus_test_providers_dependencies lotus_providers_mkldnn)
endif()

if( NOT WIN32)
    list(APPEND lotus_test_providers_libs stdc++fs)
endif()

file(GLOB_RECURSE lotus_test_tvm_src
  "${LOTUS_ROOT}/test/tvm/*.h"
  "${LOTUS_ROOT}/test/tvm/*.cc"
  )

set(lotus_test_tvm_libs
  tvm
  nnvm_compiler
  )

set(lotus_test_tvm_dependencies
  tvm
  nnvm_compiler
  )



add_library(lotus_test_utils_for_framework ${lotus_test_utils_src})
lotus_add_include_to_target(lotus_test_utils_for_framework gtest onnx protobuf::libprotobuf)
add_dependencies(lotus_test_utils_for_framework ${lotus_EXTERNAL_DEPENDENCIES})
target_include_directories(lotus_test_utils_for_framework PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE ${eigen_INCLUDE_DIRS})
# Add the define for conditionally using the framework Environment class in TestEnvironment
target_compile_definitions(lotus_test_utils_for_framework PUBLIC -DHAVE_FRAMEWORK_LIB)

if (SingleUnitTestProject)
  add_library(lotus_test_utils ALIAS lotus_test_utils_for_framework)
else()
  add_library(lotus_test_utils ${lotus_test_utils_src})
  lotus_add_include_to_target(lotus_test_utils gtest onnx protobuf::libprotobuf)
  add_dependencies(lotus_test_utils ${lotus_EXTERNAL_DEPENDENCIES})
  target_include_directories(lotus_test_utils PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE ${eigen_INCLUDE_DIRS})
endif()


if (SingleUnitTestProject)
  set(all_tests ${lotus_test_common_src} ${lotus_test_ir_src} ${lotus_test_framework_src} ${lotus_test_providers_src})
  set(all_libs lotus_test_utils ${lotus_test_providers_libs})
  set(all_dependencies ${lotus_test_providers_dependencies} )

  if (lotus_USE_TVM)
    list(APPEND all_tests ${lotus_test_tvm_src})
    list(APPEND all_libs ${lotus_test_tvm_libs})
    list(APPEND all_dependencies ${lotus_test_tvm_dependencies})
  endif()
  # we can only have one 'main', so remove them all and add back the providers test_main as it sets
  # up everything we need for all tests
  file(GLOB_RECURSE test_mains "${TEST_SRC_DIR}/*/test_main.cc")
  list(REMOVE_ITEM all_tests ${test_mains})
  list(APPEND all_tests "${TEST_SRC_DIR}/providers/test_main.cc")

  # this is only added to lotus_test_framework_libs above, but we use lotus_test_providers_libs for the lotus_test_all target.
  # for now, add it here. better is probably to have lotus_test_providers_libs use the full lotus_test_framework_libs
  # list given it's built on top of that library and needs all the same dependencies.
  if(WIN32)
    list(APPEND lotus_test_providers_libs Advapi32)
  endif()

  AddTest(
    TARGET lotus_test_all
    SOURCES ${all_tests}
    LIBS ${all_libs} ${lotus_test_common_libs}
    DEPENDS ${all_dependencies}
  )

  # the default logger tests conflict with the need to have an overall default logger
  # so skip in this type of
  target_compile_definitions(lotus_test_all PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)

  set(test_data_target lotus_test_all)
else()
  AddTest(
    TARGET lotus_test_common
    SOURCES ${lotus_test_common_src}
    LIBS ${lotus_test_common_libs}
    DEPENDS ${lotus_EXTERNAL_DEPENDENCIES}
  )

  AddTest(
    TARGET lotus_test_ir
    SOURCES ${lotus_test_ir_src}
    LIBS ${lotus_test_ir_libs}
    DEPENDS ${lotus_EXTERNAL_DEPENDENCIES}
  )

  AddTest(
    TARGET lotus_test_framework
    SOURCES ${lotus_test_framework_src}
    LIBS ${lotus_test_framework_libs}
    # code smell! see if CPUExecutionProvider should move to framework so lotus_providers isn't needed.
    DEPENDS ${lotus_test_providers_dependencies}
  )

  AddTest(
    TARGET lotus_test_providers
    SOURCES ${lotus_test_providers_src}
    LIBS ${lotus_test_providers_libs}
    DEPENDS ${lotus_test_providers_dependencies}
  )

  set(test_data_target lotus_test_ir)
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
  set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs} ${onnx_test_runner_src_dir}/lotus_event.h ${onnx_test_runner_src_dir}/simple_thread_pool.h ${onnx_test_runner_src_dir}/sync_api_linux.cc)
  set(FS_STDLIB stdc++fs)
endif()

add_library(onnx_test_runner_common ${onnx_test_runner_common_srcs})
lotus_add_include_to_target(onnx_test_runner_common lotus_test_utils onnx protobuf::libprotobuf)
add_dependencies(onnx_test_runner_common eigen ${lotus_EXTERNAL_DEPENDENCIES})
target_include_directories(onnx_test_runner_common PRIVATE ${eigen_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx)
set_target_properties(onnx_test_runner_common PROPERTIES FOLDER "LotusTest")

lotus_protobuf_generate(APPEND_PATH IMPORT_DIRS ${LOTUS_ROOT}/core/protobuf TARGET onnx_test_runner_common)

if(lotus_USE_CUDA)
  set(onnx_cuda_test_libs lotus_providers_cuda ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
endif()

if(lotus_USE_MKLDNN)
  set(onnx_mkldnn_test_libs lotus_providers_mkldnn)
endif()

set(onnx_test_libs
  lotus_test_utils
  lotus_session
  ${onnx_cuda_test_libs}
  ${onnx_mkldnn_test_libs}
  lotus_providers
  lotus_framework
  lotus_util
  lotusIR_graph
  onnx
  onnx_proto
  lotus_common
  protobuf::libprotobuf
  ${MLAS_LIBRARY}
  ${FS_STDLIB}
  ${CMAKE_THREAD_LIBS_INIT}
)

if (lotus_USE_OPENBLAS)
  if (WIN32)
    list(APPEND onnx_test_libs ${lotus_OPENBLAS_HOME}/lib/libopenblas.lib)
  else()
    list(APPEND onnx_test_libs openblas)
  endif()
endif()

if (lotus_USE_MKLDNN)
  list(APPEND onnx_test_libs mkldnn)
  add_custom_command(
    TARGET ${test_data_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_LIB_DIR}/${MKLDNN_SHARED_LIB} $<TARGET_FILE_DIR:${test_data_target}>
    )
endif()

if (lotus_USE_MKLML)
  add_custom_command(
    TARGET ${test_data_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${MKLDNN_LIB_DIR}/${MKLML_SHARED_LIB} ${MKLDNN_LIB_DIR}/${IOMP5MD_SHARED_LIB}
    $<TARGET_FILE_DIR:${test_data_target}>
    )
endif()

add_executable(onnx_test_runner ${onnx_test_runner_src_dir}/main.cc)
target_link_libraries(onnx_test_runner PRIVATE onnx_test_runner_common ${onnx_test_libs} ${GETOPT_LIB})
set_target_properties(onnx_test_runner PROPERTIES FOLDER "LotusTest")

if(lotus_BUILD_BENCHMARKS)
  add_executable(lotus_benchmark ${TEST_SRC_DIR}/onnx/microbenchmark/main.cc ${TEST_SRC_DIR}/onnx/microbenchmark/modeltest.cc)
  target_include_directories(lotus_benchmark PUBLIC ${lotusIR_graph_header} benchmark)
  target_compile_options(lotus_benchmark PRIVATE "/wd4141")
  if (lotus_USE_MLAS AND WIN32)
    target_include_directories(lotus_benchmark PRIVATE ${MLAS_INC})
  endif()
  target_link_libraries(lotus_benchmark PRIVATE ${onnx_test_libs} onnx_test_runner_common benchmark)
  add_dependencies(lotus_benchmark ${lotus_EXTERNAL_DEPENDENCIES})
  set_target_properties(lotus_benchmark PROPERTIES FOLDER "LotusTest")
endif()

if(WIN32)
  set(DISABLED_WARNINGS_FOR_PROTOBUF "/wd4125" "/wd4456" "/wd4505")
  target_compile_options(onnx_test_runner_common PRIVATE ${DISABLED_WARNINGS_FOR_PROTOBUF} -D_CRT_SECURE_NO_WARNINGS)
  target_compile_options(onnx_test_runner PRIVATE ${DISABLED_WARNINGS_FOR_PROTOBUF})
  #Maybe "CMAKE_SYSTEM_PROCESSOR" is better
  if(NOT ${CMAKE_GENERATOR_PLATFORM} MATCHES "ARM")
    add_library(onnx_test_runner_vstest SHARED ${onnx_test_runner_src_dir}/vstest_logger.cc ${onnx_test_runner_src_dir}/vstest_main.cc)
    target_compile_options(onnx_test_runner_vstest PRIVATE ${DISABLED_WARNINGS_FOR_PROTOBUF})
    target_include_directories(onnx_test_runner_vstest PRIVATE ${date_INCLUDE_DIR})
    target_link_libraries(onnx_test_runner_vstest PRIVATE ${onnx_test_libs} onnx_test_runner_common)
    set_target_properties(onnx_test_runner_vstest PROPERTIES FOLDER "LotusTest")
  endif()
endif()

set(lotus_exec_src_dir ${TEST_SRC_DIR}/lotus_exec)
file(GLOB lotus_exec_src
  "${lotus_exec_src_dir}/*.cc"
  "${lotus_exec_src_dir}/*.h"
  )
add_executable(lotus_exec ${lotus_exec_src})
# we need to force these dependencies to build first. just using target_link_libraries isn't sufficient
add_dependencies(lotus_exec ${lotus_EXTERNAL_DEPENDENCIES})
target_link_libraries(lotus_exec PRIVATE ${onnx_test_libs})
set_target_properties(lotus_exec PROPERTIES FOLDER "LotusTest")

add_test(NAME onnx_test_pytorch_converted
  COMMAND onnx_test_runner ${PROJECT_SOURCE_DIR}/external/onnx/onnx/backend/test/data/pytorch-converted)
add_test(NAME onnx_test_pytorch_operator
  COMMAND onnx_test_runner ${PROJECT_SOURCE_DIR}/external/onnx/onnx/backend/test/data/pytorch-operator)

set(lotus_perf_test_src_dir ${TEST_SRC_DIR}/perftest)
set(lotus_perf_test_src_patterns
  "${lotus_perf_test_src_dir}/*.cc"
  "${lotus_perf_test_src_dir}/*.h")

if(WIN32)
  list(APPEND lotus_perf_test_src_patterns
    "${lotus_perf_test_src_dir}/windows/*.cc"
    "${lotus_perf_test_src_dir}/windows/*.h" )
else ()
  list(APPEND lotus_perf_test_src_patterns
    "${lotus_perf_test_src_dir}/posix/*.cc"
    "${lotus_perf_test_src_dir}/posix/*.h" )
endif()

file(GLOB lotus_perf_test_src ${lotus_perf_test_src_patterns})

add_executable(lotus_perf_test ${lotus_perf_test_src})
target_include_directories(lotus_perf_test PUBLIC ${lotusIR_graph_header} ${onnx_test_runner_src_dir} ${lotus_exec_src_dir})

target_link_libraries(lotus_perf_test PRIVATE onnx_test_runner_common ${onnx_test_libs} ${GETOPT_LIB})
set_target_properties(lotus_perf_test PROPERTIES FOLDER "LotusTest")

# shared lib
if (lotus_BUILD_SHARED_LIB)
if (UNIX)
  # test custom op shared lib
  file(GLOB lotus_custom_op_shared_lib_test_srcs "${LOTUS_ROOT}/test/custom_op_shared_lib/test_custom_op.cc")

  add_library(lotus_custom_op_shared_lib_test SHARED ${lotus_custom_op_shared_lib_test_srcs})
  target_include_directories(lotus_custom_op_shared_lib_test PUBLIC "${PROJECT_SOURCE_DIR}/include")

  target_link_libraries(lotus_custom_op_shared_lib_test
    lotus_runtime
    )
  set_target_properties(lotus_custom_op_shared_lib_test PROPERTIES FOLDER "LotusSharedLibTest")

  #################################################################
  # test inference using shared lib + custom op
  file(GLOB lotus_shared_lib_test_srcs "${LOTUS_ROOT}/test/shared_lib/test_inference.cc")

  add_executable(lotus_shared_lib_test ${lotus_shared_lib_test_srcs})
  target_include_directories(lotus_shared_lib_test PUBLIC "${PROJECT_SOURCE_DIR}/include")

  target_link_libraries(lotus_shared_lib_test
    lotus_runtime
    onnx
    onnx_proto
    )
  set_target_properties(lotus_shared_lib_test PROPERTIES LINK_FLAGS "-Wl,-rpath,\$ORIGIN")  
  set_target_properties(lotus_shared_lib_test PROPERTIES FOLDER "LotusSharedLibTest")
endif()
endif()



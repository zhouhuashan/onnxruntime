find_package(Threads)

function(add_whole_archive_flag lib output_var)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(${output_var} -Wl,-force_load,$<TARGET_FILE:${lib}> PARENT_SCOPE)
  elseif(MSVC)
    # In MSVC, we will add whole archive in default.
    set(${output_var} -WHOLEARCHIVE:$<SHELL_PATH:$<TARGET_FILE:${lib}>> PARENT_SCOPE)
  else()
    # Assume everything else is like gcc
    set(${output_var} "-Wl,--whole-archive $<TARGET_FILE:${lib}> -Wl,--no-whole-archive" PARENT_SCOPE)
  endif()
endfunction()
  
function(AddTest)
  cmake_parse_arguments(_UT "" "TARGET" "LIBS;SOURCES;DEPENDS" ${ARGN})

  list(REMOVE_DUPLICATES _UT_LIBS)
  list(REMOVE_DUPLICATES _UT_SOURCES)
  
  if (lotus_RUN_ONNX_TESTS)
    list(APPEND _UT_DEPENDS models)
  endif()
  
  if (_UT_DEPENDS)
    list(REMOVE_DUPLICATES _UT_DEPENDS)
  endif(_UT_DEPENDS)
    
  add_executable(${_UT_TARGET} ${_UT_SOURCES})
  source_group(TREE ${LOTUS_ROOT}/test FILES ${_UT_SOURCES})
  set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "LotusTest")

  add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  target_include_directories(${_UT_TARGET} PUBLIC ${googletest_INCLUDE_DIRS} ${lotusIR_graph_header})
  target_link_libraries(${_UT_TARGET} ${_UT_LIBS} ${CMAKE_THREAD_LIBS_INIT})
  if (WIN32)
    target_compile_options(${_UT_TARGET} PRIVATE
        /EHsc   # exception handling - C++ may throw, extern "C" will not
    )
  endif()

  # Add the define for conditionally using the framework Environment class in TestEnvironment
  if (${lotus_framework_whole_archive} IN_LIST _UT_LIBS)
    # message("Adding -DHAVE_FRAMEWORK_LIB for " ${_UT_TARGET})
    target_compile_definitions(${_UT_TARGET} PUBLIC -DHAVE_FRAMEWORK_LIB)
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

add_whole_archive_flag(lotus_framework lotus_framework_whole_archive)
add_whole_archive_flag(lotus_providers lotus_providers_whole_archive)
add_whole_archive_flag(onnx onnx_whole_archive)

file(GLOB lotus_test_utils_src
    "${LOTUS_ROOT}/test/*.h"
    "${LOTUS_ROOT}/test/*.cc"
)

# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(lotus_test_common_libs
    lotus_common
    ${googletest_STATIC_LIBRARIES}
)

file(GLOB lotus_test_common_src
    "${LOTUS_ROOT}/test/common/*.cc"
    "${LOTUS_ROOT}/test/common/*.h"
    "${LOTUS_ROOT}/test/common/logging/*.cc"
    "${LOTUS_ROOT}/test/common/logging/*.h"
)

AddTest(
    TARGET lotus_test_common
    SOURCES ${lotus_test_utils_src} ${lotus_test_common_src}
    LIBS ${lotus_test_common_libs}
    DEPENDS googletest 
)

set(lotus_test_ir_libs
    lotusIR_graph
    ${onnx_whole_archive}
    lotus_common
    ${protobuf_STATIC_LIBRARIES}
    ${googletest_STATIC_LIBRARIES}
)

file(GLOB lotus_test_ir_src
    "${LOTUS_ROOT}/test/ir/*.cc"
    "${LOTUS_ROOT}/test/ir/*.h"
)

AddTest(
    TARGET lotus_test_ir
    SOURCES ${lotus_test_utils_src} ${lotus_test_ir_src}
    LIBS ${lotus_test_ir_libs}
    DEPENDS lotusIR_graph googletest
)

set(lotus_test_framework_libs
    ${lotus_providers_whole_archive} # code smell! see if CPUExecutionProvider should move to framework so this isn't needed.
    ${lotus_framework_whole_archive}
    lotusIR_graph
    ${onnx_whole_archive}
    lotus_common
    ${protobuf_STATIC_LIBRARIES}
    ${googletest_STATIC_LIBRARIES}
)
    
set(lotus_test_framework_src_patterns
    "${LOTUS_ROOT}/test/framework/*.cc"
    "${LOTUS_ROOT}/test/platform/*.cc"
    "${LOTUS_ROOT}/test/lib/*.cc"
)

if(WIN32)
    list(APPEND lotus_test_framework_src_patterns
         "${LOTUS_ROOT}/test/platform/windows/logging/*.h"
         "${LOTUS_ROOT}/test/platform/windows/logging/*.cc" )
endif()

file(GLOB lotus_test_framework_src ${lotus_test_framework_src_patterns})

AddTest(
    TARGET lotus_test_framework
    SOURCES ${lotus_test_utils_src} ${lotus_test_framework_src}
    LIBS ${lotus_test_framework_libs}
    # code smell! see if CPUExecutionProvider should move to framework so lotus_providers isn't needed.
    DEPENDS lotus_framework lotus_providers googletest
)

set(lotus_test_providers_libs
    ${lotus_providers_whole_archive}
    ${lotus_framework_whole_archive}
    lotusIR_graph
    ${onnx_whole_archive}
    lotus_common
    ${protobuf_STATIC_LIBRARIES}
    ${googletest_STATIC_LIBRARIES}
)

file(GLOB_RECURSE lotus_test_providers_src
    "${LOTUS_ROOT}/test/providers/*.h"
    "${LOTUS_ROOT}/test/providers/*.cc"
)

AddTest(
    TARGET lotus_test_providers
    SOURCES ${lotus_test_utils_src} ${lotus_test_providers_src}
    LIBS ${lotus_test_providers_libs}
  DEPENDS lotus_providers googletest
)

#
# LotusIR_graph test data
#
set(TEST_DATA_SRC ${LOTUS_ROOT}/test/testdata)
set(TEST_DATA_DES $<TARGET_FILE_DIR:lotus_test_ir>/testdata)

# Copy test data from source to destination.
add_custom_command(
    TARGET lotus_test_ir POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${TEST_DATA_SRC}
            ${TEST_DATA_DES})

# Copy large onnx models to test dir
if (lotus_RUN_ONNX_TESTS)
  add_custom_command(
      TARGET lotus_test_ir POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              ${CMAKE_CURRENT_BINARY_DIR}/models/models/onnx
              $<TARGET_FILE_DIR:lotus_test_ir>/models
      DEPENDS lotus_test_ir)
endif()

set(onnx_test_runner_src_dir ${LOTUS_ROOT}/test/onnx)
set(onnx_test_runner_srcs ${onnx_test_runner_src_dir}/TestCaseInfo.h
${onnx_test_runner_src_dir}/TestResultStat.cc
${onnx_test_runner_src_dir}/TestResultStat.h
${onnx_test_runner_src_dir}/main.cc
${onnx_test_runner_src_dir}/testenv.h
${onnx_test_runner_src_dir}/FixedCountFinishCallback.h
${onnx_test_runner_src_dir}/IFinishCallback.h
${onnx_test_runner_src_dir}/testenv.cc
)
if(WIN32)
  set(onnx_test_runner_srcs ${onnx_test_runner_srcs} ${onnx_test_runner_src_dir}/getopt.cc ${onnx_test_runner_src_dir}/getopt.h ${onnx_test_runner_src_dir}/parallel_runner_win.cc)
else()
  set(onnx_test_runner_srcs ${onnx_test_runner_srcs} )
  set(FS_STDLIB stdc++fs)
endif()
add_executable(onnx_test_runner ${onnx_test_runner_srcs})
target_include_directories(onnx_test_runner PUBLIC ${lotusIR_graph_header})
add_dependencies(onnx_test_runner lotus_providers lotus_framework lotusIR_graph onnx)
target_link_libraries(onnx_test_runner ${FS_STDLIB} ${lotus_providers_whole_archive} ${lotus_framework_whole_archive} lotusIR_graph ${onnx_whole_archive} lotus_common ${protobuf_STATIC_LIBRARIES}  ${CMAKE_THREAD_LIBS_INIT})

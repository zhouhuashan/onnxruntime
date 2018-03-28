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

add_whole_archive_flag(lotus_common lotus_common_whole_archive)
add_whole_archive_flag(lotusIR_graph lotusIR_graph_whole_archive)
add_whole_archive_flag(lotus_framework lotus_framework_whole_archive)
add_whole_archive_flag(lotus_providers lotus_providers_whole_archive)
add_whole_archive_flag(onnx onnx_whole_archive)

# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(lotus_test_common_libs
    ${lotus_common_whole_archive}
    ${protobuf_STATIC_LIBRARIES}
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
    SOURCES ${lotus_test_common_src}
    LIBS ${lotus_test_common_libs}
    DEPENDS googletest 
)

set(lotus_test_ir_libs
    ${lotusIR_graph_whole_archive}
    ${onnx_whole_archive}
    ${lotus_common_whole_archive}
    ${protobuf_STATIC_LIBRARIES}
    ${googletest_STATIC_LIBRARIES}
)

file(GLOB lotus_test_ir_src
    "${LOTUS_ROOT}/test/ir/*.cc"
)

AddTest(
    TARGET lotus_test_ir
    SOURCES ${lotus_test_ir_src}
    LIBS ${lotus_test_ir_libs}
    DEPENDS googletest lotusIR_graph 
)

set(lotus_test_framework_libs
    ${lotus_providers_whole_archive}
    ${lotus_framework_whole_archive}
    ${lotusIR_graph_whole_archive}
    ${onnx_whole_archive}
    ${lotus_common_whole_archive}
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
    SOURCES ${lotus_test_framework_src}
    LIBS ${lotus_test_framework_libs}
    DEPENDS lotus_framework lotus_providers googletest
)

set(lotus_test_providers_libs
    ${lotus_providers_whole_archive}
    ${lotus_framework_whole_archive}
    ${lotusIR_graph_whole_archive}
    ${onnx_whole_archive}
    ${lotus_common_whole_archive}
    ${protobuf_STATIC_LIBRARIES}
    ${googletest_STATIC_LIBRARIES}
)

file(GLOB_RECURSE lotus_test_providers_src
    "${LOTUS_ROOT}/test/providers/*.cc"
)

file(GLOB lotus_test_providers_helpers_src
    "${LOTUS_ROOT}/test/framework/framework_test_main.cc"
    "${LOTUS_ROOT}/test/*.h"
    "${LOTUS_ROOT}/test/*.cc"
)

AddTest(
    TARGET lotus_test_providers
    SOURCES ${lotus_test_providers_src} ${lotus_test_providers_helpers_src}
    LIBS ${lotus_test_providers_libs}
  DEPENDS lotus_providers googletest
)

file(GLOB_RECURSE lotus_model_tests_src
    "${LOTUS_ROOT}/test/framework/framework_test_main.cc"
    "${LOTUS_ROOT}/test/model_test/*.cc"
)

set(lotus_model_tests_libs
    ${lotus_providers_whole_archive}
    ${lotus_framework_whole_archive}
    ${lotusIR_graph_whole_archive}
    ${onnx_whole_archive}
    ${lotus_common_whole_archive}
    ${protobuf_STATIC_LIBRARIES}
    ${googletest_STATIC_LIBRARIES}
)

AddTest(
    TARGET lotus_model_tests
    SOURCES ${lotus_model_tests_src}
    LIBS ${lotus_model_tests_libs}
    DEPENDS lotus_providers lotus_framework googletest
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

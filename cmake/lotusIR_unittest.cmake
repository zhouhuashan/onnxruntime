set(UT_NAME ${PROJECT_NAME}_UT)

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

add_whole_archive_flag(lotusIR_graph lotusIR_graph_whole_archived)
add_whole_archive_flag(onnx onnx_whole_archived)
add_whole_archive_flag(lotus_provider lotus_provider_whole_archived)

set(${UT_NAME}_libs
    ${googletest_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
	${lotusIR_graph_whole_archived}
	${onnx_whole_archived}
)

file(GLOB_RECURSE ${UT_NAME}_src
    "${LOTUS_ROOT}/test/ir/*.cc"
)

AddTest(
    TARGET ${UT_NAME}
    SOURCES ${${UT_NAME}_src}
    LIBS ${${UT_NAME}_libs}
	DEPENDS googletest lotusIR_graph
)

set(lotus_test_framework_libs
    ${googletest_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
	${lotusIR_graph_whole_archived}
	${onnx_whole_archived}
	lotus_framework
	${lotus_provider_whole_archived}
)

file(GLOB_RECURSE lotus_test_framework_src
    "${LOTUS_ROOT}/test/framework/*.cc"
    "${LOTUS_ROOT}/test/platform/*.cc"
    "${LOTUS_ROOT}/test/lib/*.cc"    
)

AddTest(
    TARGET lotus_test_framework
    SOURCES ${lotus_test_framework_src}
    LIBS ${lotus_test_framework_libs}
    DEPENDS lotus_framework googletest lotusIR_graph lotus_provider
)

set(lotus_test_kernels_libs
    ${googletest_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
    ${lotusIR_graph_whole_archived}
    ${onnx_whole_archived}
    lotus_operator_kernels
)

file(GLOB_RECURSE lotus_test_kernels_src
    "${LOTUS_ROOT}/test/framework/framework_test_main.cc"
    "${LOTUS_ROOT}/test/kernels/*.cc"
    "${LOTUS_ROOT}/test/*.h"
)

AddTest(
    TARGET lotus_test_kernels
    SOURCES ${lotus_test_kernels_src}
    LIBS ${lotus_test_kernels_libs}
  DEPENDS lotus_operator_kernels lotus_framework googletest lotusIR_graph
)


set(TEST_DATA_SRC ${LOTUS_ROOT}/test/testdata)
set(TEST_DATA_DES $<TARGET_FILE_DIR:${UT_NAME}>/testdata)

# Copy test data from source to destination.
add_custom_command(
    TARGET ${UT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${TEST_DATA_SRC}
            ${TEST_DATA_DES})

# Copy large onnx models to test dir
if (lotus_RUN_ONNX_TESTS)
  add_custom_command(
      TARGET ${UT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              ${CMAKE_CURRENT_BINARY_DIR}/models/models/onnx
              $<TARGET_FILE_DIR:${UT_NAME}>/models
      DEPENDS ${UT_NAME})
endif()

set(UT_NAME ${PROJECT_NAME}_UT)

find_package(Threads)

function(add_whole_archive_flag lib output_var)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(${output_var} -Wl,-force_load,$<TARGET_FILE:${lib}> PARENT_SCOPE)
  elseif(MSVC)
    # In MSVC, we will add whole archive in default.
    set(${output_var} -WHOLEARCHIVE:$<TARGET_FILE:${lib}> PARENT_SCOPE)
  else()
    # Assume everything else is like gcc
    set(${output_var} -Wl,--whole-archive ${lib} -Wl,--no-whole-archive PARENT_SCOPE)
  endif()
endfunction()

set(${UT_NAME}_libs
    lotusIR_protos
    ${googletest_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
)

add_whole_archive_flag(lotusIR_graph tmp)
list(APPEND ${UT_NAME}_libs ${tmp})

file(GLOB_RECURSE ${UT_NAME}_src
    "${LOTUSIR_ROOT}/test/*.cc"
)
  
function(AddTest)
  cmake_parse_arguments(_UT "" "TARGET" "LIBS;SOURCES" ${ARGN})

  list(REMOVE_DUPLICATES _UT_LIBS)
  list(REMOVE_DUPLICATES _UT_SOURCES)
  
  add_executable(${_UT_TARGET} ${_UT_SOURCES})
  if (lotusIR_RUN_ONNX_TESTS)
    add_dependencies(${_UT_TARGET} googletest lotusIR_graph models)
  else()
    add_dependencies(${_UT_TARGET} googletest lotusIR_graph)
  endif()
  target_include_directories(${_UT_TARGET} PUBLIC ${googletest_INCLUDE_DIRS} ${lotusIR_graph_header})
  target_link_libraries(${_UT_TARGET} ${_UT_LIBS} ${CMAKE_THREAD_LIBS_INIT})

  set(TEST_ARGS)
  if (lotusIR_GENERATE_TEST_REPORTS)
    # generate a report file next to the test program
    list(APPEND TEST_ARGS
      "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>")
  endif(lotusIR_GENERATE_TEST_REPORTS)

  add_test(NAME ${_UT_TARGET}
    COMMAND ${_UT_TARGET} ${TEST_ARGS}
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
  )
endfunction(AddTest)



AddTest(
    TARGET ${UT_NAME}
    SOURCES ${${UT_NAME}_src}
    LIBS ${${UT_NAME}_libs}
)

set(TEST_DATA_SRC ${LOTUSIR_ROOT}/test/testdata)
set(TEST_DATA_DES $<TARGET_FILE_DIR:${UT_NAME}>/testdata)

# Copy test data from source to destination.
add_custom_command(
    TARGET ${UT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${TEST_DATA_SRC}
            ${TEST_DATA_DES})

# Copy large onnx models to test dir
if (lotusIR_RUN_ONNX_TESTS)
  add_custom_command(
      TARGET ${UT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              ${CMAKE_CURRENT_BINARY_DIR}/models/models/onnx
              $<TARGET_FILE_DIR:${UT_NAME}>/models
      DEPENDS ${UT_NAME})
endif()

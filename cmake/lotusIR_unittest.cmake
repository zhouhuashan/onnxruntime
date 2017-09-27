set(UT_NAME ${PROJECT_NAME}_UT)

set(${UT_NAME}_libs
    -WHOLEARCHIVE:$<TARGET_FILE:lotusIR_graph>
    lotusIR_protos
    ${googletest_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
)

file(GLOB_RECURSE ${UT_NAME}_src
    "${LOTUSIR_ROOT}/test/*.cc"
)
  
function(AddTest)
  cmake_parse_arguments(_UT "" "TARGET" "LIBS;SOURCES" ${ARGN})

  list(REMOVE_DUPLICATES _UT_LIBS)
  list(REMOVE_DUPLICATES _UT_SOURCES)
  
  add_executable(${_UT_TARGET} ${_UT_SOURCES})
  add_dependencies(${_UT_TARGET} googletest lotusIR_graph)
  target_include_directories(${_UT_TARGET} PUBLIC ${googletest_INCLUDE_DIRS} ${lotusIR_graph_header})
  target_link_libraries(${_UT_TARGET} ${_UT_LIBS})
  
endfunction(AddTest)

AddTest(
    TARGET ${UT_NAME}
    SOURCES ${${UT_NAME}_src}
    LIBS ${${UT_NAME}_libs}
)

set(TEST_DATA_SRC ${LOTUSIR_ROOT}/test/testdata)
set(TEST_DATA_DES ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/testdata)

#Copy test data from source to destination.
add_custom_command(
    TARGET ${UT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${TEST_DATA_SRC}
            ${TEST_DATA_DES})
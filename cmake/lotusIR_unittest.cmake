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

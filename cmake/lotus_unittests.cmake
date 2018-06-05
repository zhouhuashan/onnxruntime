find_package(Threads)

function(AddTest)
  cmake_parse_arguments(_UT "" "TARGET" "LIBS;SOURCES;DEPENDS" ${ARGN})

  list(REMOVE_DUPLICATES _UT_LIBS)
  list(REMOVE_DUPLICATES _UT_SOURCES)
  
  if (_UT_DEPENDS)
    list(REMOVE_DUPLICATES _UT_DEPENDS)
  endif(_UT_DEPENDS)
    
  add_executable(${_UT_TARGET} ${_UT_SOURCES})
  source_group(TREE ${LOTUS_ROOT}/test FILES ${_UT_SOURCES})
  set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "LotusTest")

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif(_UT_DEPENDS)

  target_link_libraries(${_UT_TARGET} ${_UT_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${lotus_EXTERNAL_LIBRARIES})
  
  if (WIN32)
    #It's cmake bug, cannot add this compile option for cuda compiler
    #(https://gitlab.kitware.com/cmake/cmake/issues/17535)
    string(APPEND CMAKE_CXX_FLAGS " /EHsc") # exception handling - C++ may throw, extern "C" will not
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
if(lotus_USE_CUDA)
  add_whole_archive_flag(lotus_providers_cuda lotus_providers_cuda_whole_archive)
endif()

file(GLOB lotus_test_utils_src
    "${LOTUS_ROOT}/test/*.h"
    "${LOTUS_ROOT}/test/*.cc"
)

file(GLOB lotus_test_common_src
    "${LOTUS_ROOT}/test/common/*.cc"
    "${LOTUS_ROOT}/test/common/*.h"
    "${LOTUS_ROOT}/test/common/logging/*.cc"
    "${LOTUS_ROOT}/test/common/logging/*.h"
)

file(GLOB lotus_test_ir_src
    "${LOTUS_ROOT}/test/ir/*.cc"
    "${LOTUS_ROOT}/test/ir/*.h"
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

if(lotus_USE_CUDA)
    list(APPEND lotus_test_framework_src_patterns  ${LOTUS_ROOT}/test/framework/cuda/*)
    list(APPEND lotus_test_framework_libs ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
endif()

file(GLOB lotus_test_framework_src ${lotus_test_framework_src_patterns})


file(GLOB_RECURSE lotus_test_providers_src
    "${LOTUS_ROOT}/test/providers/*.h"
    "${LOTUS_ROOT}/test/providers/*.cc"
    ${LOTUS_ROOT}/test/framework/TestAllocatorManager.cc
    ${LOTUS_ROOT}/test/framework/TestAllocatorManager.h
)

if(NOT lotus_USE_CUDA)
    file(GLOB_RECURSE cuda_tests "${LOTUS_ROOT}/test/providers/cuda/*")  
    list(LENGTH cuda_tests len)
    if(len GREATER 0)
        list(REMOVE_ITEM lotus_test_providers_src ${cuda_tests})
    endif()
else()
    set_source_files_properties("${LOTUS_ROOT}/test/providers/provider_test_utils.cc"
        PROPERTIES
        COMPILE_FLAGS "-DUSE_CUDA"
        )
endif()


# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(lotus_test_common_libs
    lotus_common
    gtest
    gmock
)

set(lotus_test_ir_libs
    lotusIR_graph
    ${onnx_whole_archive}
    onnx_proto
    lotus_common
    protobuf::libprotobuf
    gtest gmock
)

set(lotus_test_framework_libs
    ${lotus_providers_whole_archive} # code smell! see if CPUExecutionProvider should move to framework so this isn't needed.
    ${lotus_providers_cuda_whole_archive}
    ${lotus_framework_whole_archive}
    lotusIR_graph
    ${onnx_whole_archive}
    onnx_proto
    lotus_common
    protobuf::libprotobuf
    gtest gmock
)

if(WIN32)
    list(APPEND lotus_test_framework_libs Advapi32)
endif()
    
set(lotus_test_providers_libs
    ${lotus_providers_whole_archive}
    ${lotus_providers_cuda_whole_archive}
    ${lotus_framework_whole_archive}
    lotusIR_graph
    ${onnx_whole_archive}
    onnx_proto
    lotus_common
    protobuf::libprotobuf
    gtest gmock
)

set (lotus_test_providers_dependencies lotus_providers)

if (lotus_USE_MLAS AND WIN32)
  list(APPEND lotus_test_providers_libs mlas)
endif()

if(lotus_USE_CUDA)
    list(APPEND lotus_test_providers_dependencies lotus_providers_cuda)
    list(APPEND lotus_test_providers_libs ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
endif()
    
if (SingleUnitTestProject)
    set(all_tests ${lotus_test_utils_src} ${lotus_test_common_src} ${lotus_test_ir_src} ${lotus_test_framework_src} ${lotus_test_providers_src})

    # we can only have one 'main', so remove them all and add back the providers test_main as it sets 
    # up everything we need for all tests
    file(GLOB_RECURSE test_mains "${LOTUS_ROOT}/test/*/test_main.cc")  
    list(REMOVE_ITEM all_tests ${test_mains})
    list(APPEND all_tests "${LOTUS_ROOT}/test/providers/test_main.cc")

    # this is only added to lotus_test_framework_libs above, but we use lotus_test_providers_libs for the lotus_test_all target.
    # for now, add it here. better is probably to have lotus_test_providers_libs use the full lotus_test_framework_libs
    # list given it's built on top of that library and needs all the same dependencies.
    if(WIN32)
        list(APPEND lotus_test_providers_libs Advapi32)
    endif()

    AddTest(
        TARGET lotus_test_all
        SOURCES ${all_tests}
        LIBS ${lotus_test_providers_libs}
        DEPENDS ${lotus_test_providers_dependencies}
    )

    # the default logger tests conflict with the need to have an overall default logger
    # so skip in this type of 
    target_compile_definitions(lotus_test_all PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)

    set(test_data_target lotus_test_all)
else()
    AddTest(
        TARGET lotus_test_common
        SOURCES ${lotus_test_utils_src} ${lotus_test_common_src}
        LIBS ${lotus_test_common_libs}
    )

    AddTest(
        TARGET lotus_test_ir
        SOURCES ${lotus_test_utils_src} ${lotus_test_ir_src}
        LIBS ${lotus_test_ir_libs}
        DEPENDS lotusIR_graph
    )

    AddTest(
        TARGET lotus_test_framework
        SOURCES ${lotus_test_utils_src} ${lotus_test_framework_src}
        LIBS ${lotus_test_framework_libs}
        # code smell! see if CPUExecutionProvider should move to framework so lotus_providers isn't needed.
        DEPENDS lotus_framework lotus_providers
    )

    AddTest(
        TARGET lotus_test_providers
        SOURCES ${lotus_test_utils_src} ${lotus_test_providers_src}
        LIBS ${lotus_test_providers_libs}
        DEPENDS ${lotus_test_providers_dependencies}
    )

    set(test_data_target lotus_test_ir)
endif()  # SingleUnitTestProject

#
# LotusIR_graph test data
#
set(TEST_DATA_SRC ${LOTUS_ROOT}/test/testdata)
set(TEST_DATA_DES $<TARGET_FILE_DIR:${test_data_target}>/testdata)

# Copy test data from source to destination.
add_custom_command(
    TARGET ${test_data_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${TEST_DATA_SRC}
            ${TEST_DATA_DES})

set(onnx_test_runner_src_dir ${LOTUS_ROOT}/test/onnx)
set(onnx_test_runner_common_srcs ${onnx_test_runner_src_dir}/TestCaseInfo.h
    ${onnx_test_runner_src_dir}/TestResultStat.cc
    ${onnx_test_runner_src_dir}/TestResultStat.h
    ${onnx_test_runner_src_dir}/testenv.h
    ${onnx_test_runner_src_dir}/FixedCountFinishCallback.h
    ${onnx_test_runner_src_dir}/IFinishCallback.h
    ${onnx_test_runner_src_dir}/testenv.cc
    ${onnx_test_runner_src_dir}/runner.h
    ${onnx_test_runner_src_dir}/runner.cc
    ${LOTUS_ROOT}/test/framework/TestAllocatorManager.cc
    ${LOTUS_ROOT}/test/framework/TestAllocatorManager.h
)

if(WIN32)
  set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs} ${onnx_test_runner_src_dir}/parallel_runner_win.cc)
  add_library(win_getopt ${onnx_test_runner_src_dir}/getopt.cc ${onnx_test_runner_src_dir}/getopt.h)
  set_target_properties(win_getopt PROPERTIES FOLDER "LotusTest")
  set(GETOPT_LIB win_getopt)
else()
  set(FS_STDLIB stdc++fs)
endif()

add_library(onnx_test_runner_common ${onnx_test_runner_common_srcs})
add_dependencies(onnx_test_runner_common lotus_providers lotus_framework lotusIR_graph onnx)
set_target_properties(onnx_test_runner_common PROPERTIES FOLDER "LotusTest")

set(onnx_test_libs 
    ${FS_STDLIB}
    ${lotus_providers_whole_archive}
    ${lotus_providers_cuda_whole_archive}
    ${lotus_framework_whole_archive}
    lotusIR_graph
    ${onnx_whole_archive}
    onnx_proto
    lotus_common
    protobuf::libprotobuf
    ${CMAKE_THREAD_LIBS_INIT}
)

if (lotus_USE_MLAS AND WIN32)
  list(APPEND onnx_test_libs mlas)
endif()

if(lotus_USE_CUDA)
  set_source_files_properties("${LOTUS_ROOT}/test/onnx/runner.cc" "${LOTUS_ROOT}/test/framework/inference_session_test.cc"
    PROPERTIES
    COMPILE_FLAGS "-DUSE_CUDA"
  )
  list(APPEND onnx_test_libs lotus_providers_cuda)
  list(APPEND onnx_test_libs ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
endif()

if (lotus_USE_OPENBLAS)
  if (WIN32)
    list(APPEND onnx_test_libs ${lotus_OPENBLAS_HOME}/lib/libopenblas.lib)
  else()
    list(APPEND onnx_test_libs openblas)
  endif()
endif()

add_executable(onnx_test_runner ${onnx_test_runner_src_dir}/main.cc)
target_include_directories(onnx_test_runner PUBLIC ${lotusIR_graph_header})
target_link_libraries(onnx_test_runner ${onnx_test_libs} onnx_test_runner_common ${GETOPT_LIB})
set_target_properties(onnx_test_runner PROPERTIES FOLDER "LotusTest")

if(WIN32)
  #Maybe "CMAKE_SYSTEM_PROCESSOR" is better
  if(NOT ${CMAKE_GENERATOR_PLATFORM} MATCHES "ARM")
    add_library(onnx_test_runner_vstest SHARED ${onnx_test_runner_src_dir}/vstest_logger.cc ${onnx_test_runner_src_dir}/vstest_main.cc)
    target_link_libraries(onnx_test_runner_vstest ${onnx_test_libs} onnx_test_runner_common)
    set_target_properties(onnx_test_runner_vstest PROPERTIES FOLDER "LotusTest")
  endif()
endif()

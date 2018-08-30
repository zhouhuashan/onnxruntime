find_package(PythonInterp 3.5 REQUIRED)
include(pybind11)
FIND_PACKAGE(PythonLibs 3.5 REQUIRED)
FIND_PACKAGE(NumPy)

if(NOT PYTHON_INCLUDE_DIR)
  set(PYTHON_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\""
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    RETURN_VALUE PYTHON_NOT_FOUND)
  if(${PYTHON_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get Python include directory. Is distutils installed?")
  endif(${PYTHON_NOT_FOUND})
endif(NOT PYTHON_INCLUDE_DIR)

# 2. Resolve the installed version of NumPy (for numpy/arrayobject.h).
if(NOT NUMPY_INCLUDE_DIR)
  set(NUMPY_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import numpy; print(numpy.get_include())\""
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)
  if(${NUMPY_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get NumPy include directory: Is NumPy installed?")
  endif(${NUMPY_NOT_FOUND})
endif(NOT NUMPY_INCLUDE_DIR)

# ---[ Python + Numpy
set(lotus_pybind_srcs
    ${LOTUS_ROOT}/python/lotus_pybind_state.cc
)

#TODO(@chasun): enable cuda and test it
add_library(lotus_pybind11_state MODULE ${lotus_pybind_srcs})
if(HAS_CAST_FUNCTION_TYPE)
target_compile_options(lotus_pybind11_state PRIVATE "-Wno-cast-function-type")
endif()
target_include_directories(lotus_pybind11_state PRIVATE ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})
target_include_directories(lotus_pybind11_state PRIVATE ${pybind11_INCLUDE_DIRS})
set(lotus_pybind11_state_libs
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
)

set(lotus_pybind11_state_dependencies
    ${lotus_EXTERNAL_DEPENDENCIES}
    pybind11
)

add_dependencies(lotus_pybind11_state ${lotus_pybind11_state_dependencies})
if (MSVC)
  # if MSVC, pybind11 looks for release version of python lib (pybind11/detail/common.h undefs _DEBUG)
  target_link_libraries(lotus_pybind11_state ${lotus_pybind11_state_libs} ${MLAS_LIBRARY} ${lotus_EXTERNAL_LIBRARIES} ${PYTHON_LIBRARY_RELEASE})
else()
  target_link_libraries(lotus_pybind11_state ${lotus_pybind11_state_libs} ${lotus_EXTERNAL_LIBRARIES} ${PYTHON_LIBRARY})
  set_target_properties(lotus_pybind11_state PROPERTIES LINK_FLAGS "-Wl,-rpath,\$ORIGIN")
endif()

set_target_properties(lotus_pybind11_state PROPERTIES PREFIX "")
set_target_properties(lotus_pybind11_state PROPERTIES FOLDER "Lotus")

if (MSVC)
  set_target_properties(lotus_pybind11_state PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(lotus_pybind11_state PROPERTIES SUFFIX ".so")
endif()

file(GLOB lotus_python_srcs
    "${LOTUS_ROOT}/python/*.py"
)
file(GLOB lotus_python_test_srcs
    "${LOTUS_ROOT}/test/python/*.py"
)
file(GLOB lotus_python_tools_srcs
    "${LOTUS_ROOT}/python/tools/*.py"
)
file(GLOB lotus_python_datasets_srcs
    "${LOTUS_ROOT}/python/datasets/*.py"
)
file(GLOB lotus_python_datasets_data
    "${LOTUS_ROOT}/python/datasets/*.pb"
)

# adjust based on what target/s lotus_unittests.cmake created
if (SingleUnitTestProject)
  set(test_data_target lotus_test_all)
else()
  set(test_data_target lotus_test_ir)
endif()

add_custom_command(
  TARGET lotus_pybind11_state POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${test_data_target}>/lotus/python/tools
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${test_data_target}>/lotus/python/datasets
  COMMAND ${CMAKE_COMMAND} -E copy
      ${LOTUS_ROOT}/__init__.py
      $<TARGET_FILE_DIR:${test_data_target}>/lotus/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${lotus_python_test_srcs}
      $<TARGET_FILE_DIR:${test_data_target}>
  COMMAND ${CMAKE_COMMAND} -E copy
      ${lotus_python_srcs}
      $<TARGET_FILE_DIR:${test_data_target}>/lotus/python/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${lotus_python_tools_srcs}
      $<TARGET_FILE_DIR:${test_data_target}>/lotus/python/tools/
  COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE:lotus_pybind11_state>
      $<TARGET_FILE_DIR:${test_data_target}>/lotus/python/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${lotus_python_datasets_srcs}
      $<TARGET_FILE_DIR:${test_data_target}>/lotus/python/datasets/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${lotus_python_datasets_data}
      $<TARGET_FILE_DIR:${test_data_target}>/lotus/python/datasets/
)

if (lotus_USE_MKLDNN)
  add_custom_command(
    TARGET lotus_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${MKLDNN_LIB_DIR}/${MKLDNN_SHARED_LIB}
        $<TARGET_FILE_DIR:${test_data_target}>/lotus/python/
  )
endif()
if (lotus_USE_MKLML)
  add_custom_command(
    TARGET lotus_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${MKLDNN_LIB_DIR}/${MKLML_SHARED_LIB} ${MKLDNN_LIB_DIR}/${IOMP5MD_SHARED_LIB}
        $<TARGET_FILE_DIR:${test_data_target}>/lotus/python/
  )
endif()

find_package(PythonInterp REQUIRED)
include(pybind11)

FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(NumPy)

# ---[ Python + Numpy
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIRS})


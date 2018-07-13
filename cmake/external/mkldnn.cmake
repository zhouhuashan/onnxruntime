include (ExternalProject)

set(MKLDNN_URL https://github.com/intel/mkl-dnn.git)
set(MKLDNN_TAG v0.15)
set(MKLDNN_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/mkl-dnn/src/mkl-dnn/src)
set(MKLDNN_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/mkl-dnn/install)
set(MKLDNN_LIB_DIR ${MKLDNN_INSTALL}/lib)
set(MKLDNN_INCLUDE_DIR ${MKLDNN_INSTALL}/include)
if(WIN32)
  set(MKLDNN_SHARED_LIB mkldnn.dll)
  set(MKLDNN_IMPORT_LIB mkldnn.lib)
else()
  set(MKLDNN_SHARED_LIB libmkldnn.so.0)
endif()

ExternalProject_Add(project_mkldnn
    PREFIX mkl-dnn
    GIT_REPOSITORY ${MKLDNN_URL}
    GIT_TAG ${MKLDNN_TAG}
    SOURCE_DIR ${MKLDNN_SOURCE}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL}
)

if(WIN32)
  add_library(mkldnn STATIC IMPORTED)
  set_property(TARGET mkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB_DIR}/${MKLDNN_IMPORT_LIB})
else()
  add_library(mkldnn SHARED IMPORTED)
  set_property(TARGET mkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB_DIR}/${MKLDNN_SHARED_LIB})
endif()
add_dependencies(mkldnn project_mkldnn)
include_directories(${MKLDNN_INCLUDE_DIR})

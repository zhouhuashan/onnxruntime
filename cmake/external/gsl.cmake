include (ExternalProject)

set(gsl_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/gsl)
set(gsl_INCLUDE_DIR ${gsl_ROOT_DIR}/include)
set(gsl_URL "https://github.com/Microsoft/GSL.git")
set(gsl_TAG "9c5e21045dd1138750868ed307110778ef4f4bc3")

find_package(Git)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "Failed to find Git!")
endif()

ExternalProject_Add(gsl
    PREFIX gsl
    GIT_REPOSITORY ${gsl_URL}
    GIT_TAG ${gsl_TAG}
    #https://github.com/Microsoft/GSL/issues/525
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${gsl_ROOT_DIR} -DGSL_TEST=OFF
    )

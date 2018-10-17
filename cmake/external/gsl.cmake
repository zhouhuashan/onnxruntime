include (ExternalProject)

set(gsl_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/gsl)
set(gsl_INCLUDE_DIR ${gsl_ROOT_DIR}/include)
set(gsl_URL "https://github.com/Microsoft/GSL.git")
# using TAG of "v2.0.0" would be better, but matching WinML for now.
# they are on a previous version that has the explicit not_null ctor but lacks make_not_null
set(gsl_TAG "cee3125af7208258d024a75e24f73977eddaec5b")

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

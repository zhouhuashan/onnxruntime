include (ExternalProject)

set(date_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/date)
set(date_INCLUDE_DIR ${date_ROOT_DIR}/include)
set(date_URL "https://github.com/HowardHinnant/date.git")
# NOTE: Use v2.5 when released. We need a bugfix from the listed commit to build on VS2017 15.6
#set(date_TAG "v2.4")
set(date_TAG "a1ceec19fed2196ff4e22929100b221160651aa6")

find_package(Git)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "Failed to find Git!")
endif()

ExternalProject_Add(date
    PREFIX date
    GIT_REPOSITORY ${date_URL}
    GIT_TAG ${date_TAG}
    CMAKE_CACHE_ARGS -DENABLE_DATE_TESTING:BOOL=OFF -DUSE_SYSTEM_TZ_DB:BOOL=ON
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${date_ROOT_DIR}
    )

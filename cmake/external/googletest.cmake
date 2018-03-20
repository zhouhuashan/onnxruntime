# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
include (ExternalProject)

set(googletest_INCLUDE_DIRS 
        ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/include 
        ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/include)
set(googletest_URL https://github.com/google/googletest.git)
set(googletest_BUILD ${CMAKE_CURRENT_BINARY_DIR}/googletest/)
set(googletest_TAG 9bda90b7e5e08c4c37a832d0cea218aed6af6470)

if(WIN32)
  set(googletest_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/gtest/$<CONFIG>/$<IF:$<CONFIG:Debug>,gtestd.lib,gtest.lib>
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/$<CONFIG>/$<IF:$<CONFIG:Debug>,gmockd.lib,gmock.lib>)
else()
  set(googletest_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/gtest/$<IF:$<CONFIG:Debug>,libgtestd.a,libgtest.a>
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/$<IF:$<CONFIG:Debug>,libgmockd.a,libgmock.a>)
endif()

ExternalProject_Add(googletest
    PREFIX googletest
    GIT_REPOSITORY ${googletest_URL}
    GIT_TAG ${googletest_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_SYSTEM_NAME:STRING=${CMAKE_SYSTEM_NAME}
        -DCMAKE_SYSTEM_VERSION:STRING=${CMAKE_SYSTEM_VERSION}
        -DBUILD_GMOCK:BOOL=ON
        -DBUILD_GTEST:BOOL=OFF   # Built indirectly by BUILD_GMOCK
        -Dgtest_force_shared_crt:BOOL=ON
)

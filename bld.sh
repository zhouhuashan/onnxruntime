CMAKE_BUILD_TYPE=RelWithDebInfo
if [ "$1" == "" ]; then
  CMAKE_BUILD_TYPE=Debug
fi
cd cmake
cmake -H. -BLinux/$CMAKE_BUILD_TYPE -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
cmake --build Linux/$CMAKE_BUILD_TYPE -- -j4

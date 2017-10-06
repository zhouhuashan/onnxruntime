
CMAKE_BUILD_TYPE=RelWithDebInfo
if [ "$1" == "" ]; then
  CMAKE_BUILD_TYPE=Debug
fi
cd cmake
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
cmake --build build -- -j4

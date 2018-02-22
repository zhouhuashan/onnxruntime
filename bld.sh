CMAKE_BUILD_TYPE=RelWithDebInfo
if [ "$1" == "" ]; then
  CMAKE_BUILD_TYPE=Debug
fi
cd cmake
BLD_DIR=../build/Linux/$CMAKE_BUILD_TYPE
cmake -H. -B$BLD_DIR -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
cmake --build $BLD_DIR -- -j4

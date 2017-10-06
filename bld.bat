
set CMAKE_BUILD_TYPE=RelWithDebInfo
if "%1%" == "" set CMAKE_BUILD_TYPE=Debug

mkdir cmake\build
cd cmake\build

cmake ..  -A x64 -T v140 -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%
MSBuild /p:Configuration=%CMAKE_BUILD_TYPE% ALL_BUILD.vcxproj

cd ..

set PreferredToolArchitecture=x64

set CMAKE_BUILD_TYPE=RelWithDebInfo
if "%1%" == "" set CMAKE_BUILD_TYPE=Debug

mkdir cmake\Windows\%CMAKE_BUILD_TYPE%
cd cmake\Windows\%CMAKE_BUILD_TYPE%

cmake ..\..  -A x64 -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%
MSBuild /p:Configuration=%CMAKE_BUILD_TYPE% ALL_BUILD.vcxproj

cd ..\..

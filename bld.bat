set PreferredToolArchitecture=x64

set CMAKE_BUILD_TYPE=RelWithDebInfo
if "%1%" == "" set CMAKE_BUILD_TYPE=Debug

set BLD_DIR=build\Windows\%CMAKE_BUILD_TYPE%
mkdir %BLD_DIR%
cd %BLD_DIR%

cmake ..\..\..\cmake  -A x64 -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%
MSBuild /p:Configuration=%CMAKE_BUILD_TYPE% ALL_BUILD.vcxproj

cd ..\..\..

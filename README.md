# Lotus

Lotus is the runtime for ONNX. Here is the [design document](https://microsoft.sharepoint.com/:w:/t/ONNX2/EdT4SATkbt1Nv4un1JoBHrYBH65Yt3EKFGHCuo2NTAv4Fg).

# Build
Install cmake-3.10 or better from https://cmake.org/download/.

Checkout the source tree:
```
git clone --recursive https://aiinfra.visualstudio.com/_git/Lotus
cd Lotus
```

Gerneate the project files (only needed once):
```
set CMAKE_BUILD_TYPE=Debug
mkdir cmake_build
cd cmake_build
cmake ../cmake -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%
```

And build it:
```
MSBuild /p:Configuration=%CMAKE_BUILD_TYPE% ALL_BUILD.vcxproj
```

Run unit tests:
```
ctest -C %CMAKE_BUILD_TYPE%
```
`ALL_BUILD.vcxproj` will check external dependecies and takes a little longer. During development you want to
use a more specific project like `lotus_test_core_runtime.vcxproj`.

# CUDA Build
Lotus supports CUDA builds. You need to download and install `CUDA8` and `CUDNN6` from the Nvidia website.
Debug builds for CUDA builds have issues, we use RelWithDebInfo. NVidia supports only VS2015 for CUDA8.
You can build with:
```
set CMAKE_BUILD_TYPE=RelWithDebInfo
mkdir cmake_build_gpu

cmake ../cmake -A x64 -G "Visual Studio 14 2015" -T host=x64 -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -Dlotus_USE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=where_you_installed_cuda -Dlotus_CUDNN_HOME=where_you_installed_cudnn
```

# Source tree structure
TODO

Cmake will automatically generate the project files for those locations.

# Adding custom operators
TODO

# Unit Tests
We use gtest as framework for unit tests. Test in the same directory are linked
into 1 exe. More TODO. 

# Integration Tests
TODO

# Coding guidelines
[Google C++ style guide](https://google.github.io/styleguide/cppguide.html)

# Checkin procedure
```
git clone ...
git checkout -b my_changes
make your changes
git commit -m "my changes"
git push
```
To request merge into master send a pull request from the web ui
https://aiinfra.visualstudio.com/_git/Lotus
or with codeflow. New code should be accompanied by unittests.

# Additional Build Flavors
## MKL
To build Lotus with MKL support, download MKL from Intel and call cmake the following way:
```
cmake .. -G "Visual Studio 14 2015" -A x64   -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -DCMAKE_CXX_FLAGS="/openmp" -Dlotus_USE_EIGEN=OFF -Dlotus_USE_MKL=ON -Dlotus_MKL_HOME=%MKL_HOME%
```
where MKL_HOME would be something like:
`D:\local\IntelSWTools\compilers_and_libraries\windows\mkl`

## Openblas
To build Lotus with Openblas support, download Openblas and compile it for windows.
Instructions how to build Openblas for windows can be found here https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio#build-openblas-for-universal-windows-platform.

Once you have the Openblas binaries, call the Lotus cmake like:
```
cmake .. -G "Visual Studio 14 2015" -A x64   -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -DCMAKE_CXX_FLAGS="/openmp"  -Dlotus_USE_EIGEN=OFF -Dlotus_USE_OPENBLAS=ON -Dlotus_OPENBLAS_HOME=%OPENBLAS_HOME%
```
where OPENBLAS_HOME would be something like:
`d:\share\openblas`

## AVX, AVX2, OpenMP
To pass in additional compiler flags, for example to build with SIMD instructions, you can pass in CXX_FLAGS from the cmake command line, for example to build eigen with avx2 support and openmp, you can call cmake like:
```
cmake .. -G "Visual Studio 14 2015" -A x64   -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -DCMAKE_CXX_FLAGS="/arch:AVX2 /openmp"
```
## CUDA
To build Lotus with CUDA support, download CUDA9.1 and CUDNN6 from NVidia and CMake 3.11 For VisualStudio 2017, it also need VC compiler v14.11, instead of default v14.13.
Once installed you can call cmake for Lotus as follows:
```
python %~dp0\tools\ci_build\build.py --build_dir %~dp0\build\Windows --use_cuda --cudnn_home %CUDNN_HOME% %* 

you may have to specify your root CUDA installation directory via -DCUDA_TOOLKIT_ROOT_DIR
```
where CUDNN_HOME may look something like `d:\local\cudnn-8.0-windows10-x64-v6.0\cuda`
and
CUDA_TOOLKIT_ROOT_DIR is something like `c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`

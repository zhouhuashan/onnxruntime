# Lotus

Lotus is the runtime for ONNX. Here is the [design document](https://microsoft.sharepoint.com/:w:/t/ONNX2/EdT4SATkbt1Nv4un1JoBHrYBH65Yt3EKFGHCuo2NTAv4Fg).

## Supported dev environments

| OS          | Supports CPU | Supports GPU| Notes                              | 
|-------------|:------------:|:------------:|------------------------------------|
|Windows 10   | YES          | YES         |Must use VS 2017 or the latest VS2015|
|Windows 10 <br/> Subsystem for Linux | YES         | NO        |         |
|Ubuntu 16.x  | YES          | YES         |                            |
|Ubuntu 17.x  | YES          | YES         |                            |
|Ubuntu 18.x  | YES          | UNKNOWN     | No CUDA package from Nvidia|
|Fedora 24    | YES          | YES         |                            |
|Fedora 25    | YES          | YES         |                            |
|Fedora 26    | YES          | YES         |                            |
|Fedora 27    | YES          | YES         |                            |
|Fedora 28    | YES          | NO          |Cannot build GPU kernels but can run them |

Red Hat Enterprise Linux and CentOS are not supported.
Clang 7.x is not supported. You may use Clang 6.x.
GCC 4.x and below are not supported. If you are using GCC 7.0+, you'll need to upgrade eigen to a newer version before compiling Lotus.

OS/Compiler Matrix:

|             | Supports VC  | Supports GCC     |  Supports Clang |
|-------------|:------------:|:----------------:|:---------------:|
|Windows 10   | YES          | Not tested       | Not tested      |
|Linux        | NO           | YES(gcc>=5.0)    | YES             |

Lotus python binding only supports Python 3.x. You'd better use python 3.5+.

# Build
Install cmake-3.10 or better from https://cmake.org/download/.

Checkout the source tree:
```
git clone --recursive https://aiinfra.visualstudio.com/_git/Lotus
cd Lotus
```

Generate the project files (only needed once):
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

## Enable Clang tools
You may also add '-DCMAKE\_EXPORT\_COMPILE\_COMMANDS=ON' to your cmake args, then your build engine(like msbuild/make/ninja) will generate a compile\_commands.json file for you. Please copy this file to the top source directory. Then you can use clang tools like ['clang-rename'](http://clang.llvm.org/extra/clang-rename.html), ['clang-tidy'](http://clang.llvm.org/extra/clang-tidy/) to clean up or refactor your code.

# Source tree structure
TODO

Cmake will automatically generate the project files for those locations.

# Adding custom operators
TODO

# Unit Tests
We use gtest as framework for unit tests. Test in the same directory are linked
into 1 exe. More TODO. 

# Integration Tests
To run onnx model tests on Linux,

1. Install docker
2. (optional) Run "export AZURE\_BLOB\_KEY=<secret_value>". You can get the key by executing "az storage keys list --account-name lotus" if you have Azure CLI 2.0 installed or just ask chasun@microsoft.com for that.
3.  Run tools/ci\_build/vsts/linux/run\_build.sh.

For Windows, please follow the README file at lotus/test/onnx/README.txt

# Contribution guidelines
Please see [CONTRIBUTING.md]

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
## Windows CUDA Build
Lotus supports CUDA builds. You will need to download and install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [CUDNN](https://developer.nvidia.com/cudnn).

Lotus is built and tested with CUDA 9.0 and CUDNN 7.0 using the Visual Studio 2017 14.11 toolset (i.e. Visual Studio 2017 v15.3). 
CUDA versions up to 9.2 and CUDNN version 7.1 should also work with versions of Visual Studio 2017 up to and including v15.7, however you may need to explicitly install and use the 14.11 toolset due to CUDA and CUDNN only being compatible with earlier versions of Visual Studio 2017.

To install the Visual Studio 2017 14.11 toolset, see <https://blogs.msdn.microsoft.com/vcblog/2017/11/15/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2017/> 

If using this toolset with a later version of Visual Studio 2017 you have two options:

1. Setup the Visual Studio environment variables to point to the 14.11 toolset by running vcvarsall.bat prior to running cmake
   - e.g.  if you have VS2017 Enterprise, an x64 build would use the following command
`"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.11`

2. Alternatively if you have CMake 3.12 or later you can specify the toolset version in the "-T" parameter by adding "version=14.11"
   - e.g. use the following with the below cmake command
`-T version=14.11,host=x64`

CMake should automatically find the CUDA installation. If it does not, or finds a different version to the one you wish to use, specify your root CUDA installation directory via the -DCUDA_TOOLKIT_ROOT_DIR CMake parameter.  

_Side note: If you have multiple versions of CUDA installed on a Windows machine and are building with Visual Studio, CMake will use the build files for the highest version of CUDA it finds in the BuildCustomization folder.  e.g. C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\IDE\VC\VCTargets\BuildCustomizations\. If you want to build with an earlier version, you must temporarily remove the 'CUDA x.y.*' files for later versions from this directory._ 

The path to the 'cuda' folder in the CUDNN installation must be provided. The 'cuda' folder should contain 'bin', 'include' and 'lib' directories.

You can build with:

```
mkdir cmake_build_gpu
cd cmake_build_gpu    
cmake ..\cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -A x64 -G "Visual Studio 15 2017" -T host=x64 -Dlotus_USE_CUDA=ON -Dlotus_CUDNN_HOME=<path to top level 'cuda' directory in CUDNN installation>
```
	
where the CUDNN path would be something like `C:\cudnn-9.2-windows10-x64-v7.1\cuda`

## MKL
To build Lotus with MKL support, download MKL from Intel and call cmake the following way:
```
mkdir cmake_build_mkl
cd cmake_build_mkl
cmake ..\cmake -G "Visual Studio 15 2017" -A x64 -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -DCMAKE_CXX_FLAGS="/openmp" -Dlotus_USE_EIGEN=OFF -Dlotus_USE_MKL=ON -Dlotus_MKL_HOME=%MKL_HOME%
```
where MKL_HOME would be something like:
`D:\local\IntelSWTools\compilers_and_libraries\windows\mkl`

## OpenBLAS
To build Lotus with OpenBLAS support, download OpenBLAS and compile it for windows.
Instructions how to build OpenBLAS for windows can be found here https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio#build-openblas-for-universal-windows-platform.

Once you have the OpenBLAS binaries, call the Lotus cmake like:
```
mkdir cmake_build_openblas
cd cmake_build_openblas
cmake ..\cmake -G "Visual Studio 15 2017" -A x64 -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -DCMAKE_CXX_FLAGS="/openmp"  -Dlotus_USE_EIGEN=OFF -Dlotus_USE_OPENBLAS=ON -Dlotus_OPENBLAS_HOME=%OPENBLAS_HOME%
```
where OPENBLAS_HOME would be something like:
`d:\share\openblas`

For Linux (e.g. Ubuntu 16.04), install libopenblas-dev package
sudo apt-get install libopenblas-dev

## AVX, AVX2, OpenMP
To pass in additional compiler flags, for example to build with SIMD instructions, you can pass in CXX_FLAGS from the cmake command line, for example to build eigen with avx2 support and openmp, you can call cmake like:
```
cmake .. -G "Visual Studio 15 2017" -A x64   -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -DCMAKE_CXX_FLAGS="/arch:AVX2 /openmp"
```

## Build with Docker on Linux
Install Docker: `https://docs.docker.com/install/`

###CPU
```
cd tools/ci_build/vsts/linux/docker
docker build -t lotus_dev --build-arg OS_VERSION=16.04 -f Dockerfile.ubuntu .
docker run --rm -it lotus_dev /bin/bash
```

###GPU
If you need GPU support, please also install:
1. nvidia driver. Before doing this please add 'nomodeset rd.driver.blacklist=nouveau' to your linux [kernel boot parameters](https://www.kernel.org/doc/html/v4.17/admin-guide/kernel-parameters.html).
2. nvidia-docker2: [Install doc](`https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)`)

To test if your nvidia-docker works:
```
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```

Then build a docker image. We provided a sample for use:
```
cd tools/ci_build/vsts/linux/docker
docker build -t cuda_dev -f Dockerfile.ubuntu_gpu .
```

Then run it
```
cd ~/src
git clone https://aiinfra.visualstudio.com/Lotus/_git/Lotus
docker run --runtime=nvidia -v ~/src/Lotus:/data/lotus --rm -it cuda_dev /bin/bash
mkdir build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug /data/lotus/cmake -Dlotus_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Dlotus_USE_CUDA=ON -Dlotus_CUDNN_HOME=/usr/local/cudnn-7.0/cuda
ninja
```


## Build with Docker (CPU) on Windows
Register a docker account at [https://www.docker.com/](https://www.docker.com/)

Download Docker for Windows: `https://store.docker.com/editions/community/docker-ce-desktop-windows`

Install Docker for Windows. Share local drive to Docker, open Docker Settings->Shared Drives, share the disk drive to docker. This is used to mount the local code to docker instance.

Run powershell command to build docker

```
cd .\Lotus\tools\ci_build\vsts\linux\ubuntu16.04
docker login
docker build -t lotus-ubuntu16.04 .
docker run -it --rm --name lotus-cpu -v [LocalPath]/Git/Lotus:/home/lotusdev/Lotus lotus-ubuntu16.04 /bin/bash
source /usr/local/miniconda3/bin/activate lotus-py35
python /home/lotusdev/Lotus/tools/ci_build/build.py --build_dir /home/lotusdev/Lotus/build/Linux --config Debug --skip_submodule_sync --enable_pybind
```
Run command below if the conda environment `lotus-py35` does not exist
```
/usr/local/miniconda3/bin/conda env create --file /home/lotusdev/Lotus/tools/ci_build/vsts/linux/Conda/conda-linux-lotus-py35-environment.yml --name lotus-py35 --quiet --force
```

# Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# License
See [LICENSE]
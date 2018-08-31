#!/bin/bash
#install ninja
aria2c -q -d /tmp https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
unzip -oq /tmp/ninja-linux.zip -d /usr/bin
rm -f /tmp/ninja-linux.zip
#install protobuf
mkdir -p /tmp/src
mkdir -p /opt/cmake
aria2c -q -d /tmp/src   https://cmake.org/files/v3.12/cmake-3.12.1-Linux-x86_64.tar.gz
tar -xf /tmp/src/cmake-3.12.1-Linux-x86_64.tar.gz --strip 1 -C /opt/cmake
aria2c -q -d /tmp/src https://github.com/protocolbuffers/protobuf/archive/v3.6.1.tar.gz
tar -xf /tmp/src/protobuf-3.6.1.tar.gz -C /tmp/src
cd /tmp/src/protobuf-3.6.1
if [ -f /etc/redhat-release ] ; then
  PB_LIBDIR=lib64
else
  PB_LIBDIR=lib
fi
for build_type in 'Debug' 'Relwithdebinfo'; do
  pushd .
  mkdir build_$build_type
  cd build_$build_type
  /opt/cmake/bin/cmake -G Ninja ../cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=$PB_LIBDIR  -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=$build_type
  ninja
  ninja install
  popd
done
export ONNX_ML=1
#3376d4438aaadfba483399fa249b841153152bc0 is v1.2.2
#bae6333e149a59a3faa9c4d9c44974373dcf5256 is v1.3.0
for onnx_version in "3376d4438aaadfba483399fa249b841153152bc0" "bae6333e149a59a3faa9c4d9c44974373dcf5256" ; do
  aria2c -q -d /tmp/src  https://github.com/onnx/onnx/archive/$onnx_version.tar.gz
  tar -xf /tmp/src/onnx-$onnx_version.tar.gz -C /tmp/src
  cd /tmp/src/onnx-$onnx_version
  git clone https://github.com/pybind/pybind11.git third_party/pybind11
  python3 setup.py bdist_wheel
  pip3 install -q dist/*
  mkdir -p /data/onnx/$onnx_version
  backend-test-tools generate-data -o /data/onnx/$onnx_version
  pip3 uninstall -y onnx
done

aria2c -q -d /tmp/src  http://bitbucket.org/eigen/eigen/get/3.3.5.tar.bz2
tar -jxf /tmp/src/eigen-eigen-b3f3d4950030.tar.bz2 -C /usr/include
mv /usr/include/eigen-eigen-b3f3d4950030 /usr/include/eigen3

rm -rf /tmp/src
chmod 0777 /data/onnx


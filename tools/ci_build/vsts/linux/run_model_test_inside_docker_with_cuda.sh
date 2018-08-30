#!/bin/bash
#This script uses system provided eigen and protobuf, not the ones in Lotus repo.
set -e
mkdir /tmp/model_test_build
cd /tmp/model_test_build
if [ -z "$AZURE_BLOB_KEY" ]; then
  echo "AZURE_BLOB_KEY is blank"
else
  echo "Downloading test data from azure"
  mkdir -p /data/onnx
  azcopy --recursive --source:https://lotus.blob.core.windows.net/onnx-model-zoo-20180726 --destination:/data/onnx/models  --source-key:$AZURE_BLOB_KEY
fi
PATH=$PATH:/usr/local/cuda/bin /opt/cmake/bin/cmake -G Ninja -Dlotus_USE_CUDA=ON -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs -Dlotus_CUDNN_HOME=/usr/local/cudnn-7.0/cuda -DCMAKE_BUILD_TYPE=Debug -Deigen_SOURCE_PATH=/usr/include/eigen3 -Dlotus_USE_PREINSTALLED_EIGEN=ON /data/lotus/cmake
ninja
for D in /data/onnx/*; do
    if [ -d "${D}" ]; then
        echo "running tests with ${D}" 
        ./onnx_test_runner -e cuda "${D}"
    fi
done


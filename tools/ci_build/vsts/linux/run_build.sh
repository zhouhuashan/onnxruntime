#!/bin/bash
set -e -o -x

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"

while getopts c:d:x: parameter_Option
do case "${parameter_Option}"
in
d) BUILD_DEVICE=${OPTARG};;
x) BUILD_EXTR_PAR=${OPTARG};;
esac
done

if [ -z "$AZURE_BLOB_KEY" ]; then
  echo "AZURE_BLOB_KEY is blank"
else
  echo "Downloading test data from azure"
  mkdir -p /data/onnx
  azcopy --recursive --source:https://lotus.blob.core.windows.net/onnx-model-zoo-20181018 --destination:/data/onnx/models  --source-key:$AZURE_BLOB_KEY
  BUILD_EXTR_PAR="$BUILD_EXTR_PAR --enable_onnx_tests"
fi

ln -s /data/onnx/models models

if [ $BUILD_DEVICE = "gpu" ]; then
    python3 $SCRIPT_DIR/../../build.py --build_dir /home/lotusdev \
        --config Debug Release \
        --skip_submodule_sync \
        --parallel --build_shared_lib \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/local/cudnn-7.0/cuda $BUILD_EXTR_PAR
    Release/onnx_test_runner -e cuda /data/onnx
else
    python3 $SCRIPT_DIR/../../build.py --build_dir /home/lotusdev \
        --config Debug Release --build_shared_lib \
        --skip_submodule_sync \
        --enable_pybind \
        --parallel $BUILD_EXTR_PAR
    Release/onnx_test_runner /data/onnx
    Release/onnx_test_runner -e mkldnn /data/onnx
fi

#!/bin/bash
set -e -o -x

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"

while getopts c:d:x: parameter_Option
do case "${parameter_Option}"
in
c) BUILD_CONFIG=${OPTARG};;
d) BUILD_DEVICE=${OPTARG};;
x) BUILD_EXTR_PAR=${OPTARG};;
esac
done

if [ $BUILD_DEVICE = "gpu" ]; then
    python3 $SCRIPT_DIR/../../build.py --build_dir /home/lotusdev \
        --config $BUILD_CONFIG --install_onnx \
        --skip_submodule_sync \
        --parallel \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/local/cudnn-7.0/cuda $BUILD_EXTR_PAR
else
    python3 $SCRIPT_DIR/../../build.py --build_dir /home/lotusdev \
        --config $BUILD_CONFIG --install_onnx \
        --skip_submodule_sync \
        --enable_pybind \
        --parallel $BUILD_EXTR_PAR
fi


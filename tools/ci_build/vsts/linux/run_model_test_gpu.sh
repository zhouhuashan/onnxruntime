#!/bin/bash
set -e
SCRIPT=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT`
TOP_SRC_DIR=`realpath $SCRIPT_DIR/../../../../`
echo $TOP_SRC_DIR
cd $SCRIPT_DIR/docker
docker_image=ubuntu16.04-cuda9.0-cudnn7.0
docker build -t $docker_image -f Dockerfile.ubuntu_gpu .
nvidia-docker run --rm -e NVIDIA_VISIBLE_DEVICES=all -e AZURE_BLOB_KEY -v $HOME/.ccache:/root/.ccache -v $TOP_SRC_DIR:/data/lotus $docker_image /data/lotus/tools/ci_build/vsts/linux/run_model_test_inside_docker_with_cuda.sh


#!/bin/bash
set -e
SCRIPT=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT`
TOP_SRC_DIR=`realpath $SCRIPT_DIR/../../../../`
for version in '28'; do
  docker_image=fedora$version
  cd $SCRIPT_DIR/docker
  docker build -t $docker_image --build-arg OS_VERSION=$version -f Dockerfile.fedora .
  docker run --rm -e AZURE_BLOB_KEY -v $HOME/.ccache:/root/.ccache -v $TOP_SRC_DIR:/data/lotus $docker_image /data/lotus/tools/ci_build/vsts/linux/run_model_test_inside_docker.sh
done

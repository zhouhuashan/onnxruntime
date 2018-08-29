#!/bin/bash
set -e -o -x

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
SOURCE_ROOT=$SCRIPT_DIR/../../../../

while getopts c:o:d:r:x: parameter_Option
do case "${parameter_Option}"
in
#Debug,Release
c) BUILD_CONFIG=${OPTARG};;
#ubuntu16.04
o) BUILD_OS=${OPTARG};;
#cpu, gpu
d) BUILD_DEVICE=${OPTARG};;
r) BUILD_DIR=${OPTARG};;
# "--build_wheel --use_openblas"
x) BUILD_EXTR_PAR=${OPTARG};;
esac
done

EXIT_CODE=1

echo "bc=$BUILD_CONFIG bo=$BUILD_OS bd=$BUILD_DEVICE bdir=$BUILD_DIR bex=$BUILD_EXTR_PAR"

cd $SCRIPT_DIR/docker
if [ $BUILD_DEVICE = "gpu" ]; then
    IMAGE="ubuntu16.04"
    docker build -t "lotus-$IMAGE" -f Dockerfile.ubuntu_gpu .
else
    IMAGE="ubuntu16.04-cuda9.0-cudnn7.0"
    docker build -t "lotus-$IMAGE" --build-arg OS_VERSION=16.04 -f Dockerfile.ubuntu .
fi



set +e

if [ $BUILD_DEVICE = "cpu" ]; then
    docker run -h $HOSTNAME \
        --rm \
        --name "lotus-$BUILD_DEVICE" \
        --volume "$SOURCE_ROOT:$SOURCE_ROOT" \
        --volume "$BUILD_DIR:/home/lotusdev" \
        "lotus-$IMAGE" \
        /bin/bash $SCRIPT_DIR/run_build.sh \
        -c $BUILD_CONFIG -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" &
else
    nvidia-docker run -h $HOSTNAME \
        --rm \
        --name "lotus-$BUILD_DEVICE" \
        --volume "$SOURCE_ROOT:$SOURCE_ROOT" \
        --volume "$BUILD_DIR:/home/lotusdev" \
        "lotus-$IMAGE" \
        /bin/bash $SCRIPT_DIR/run_build.sh \
        -c $BUILD_CONFIG -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" &
fi
wait -n

EXIT_CODE=$?

set -e
exit $EXIT_CODE


set -e -o

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
SOURCE_ROOT=$SCRIPT_DIR/../../../../

BUILD_CONFIG=$1

cd $SCRIPT_DIR/ubuntu16.04-cuda9.0-cudnn7.0

docker build -t lotus-ubuntu16.04-cuda9.0-cudnn7.0 .

NVIDIA_DRIVER_NAME="$(docker volume ls --format "{{.Name}}" | grep -Po 'nvidia_driver_[\d.]+' | sort -V -r| head -n 1)"

if [[ -z $NVIDIA_DRIVER_NAME ]]; then
    echo "No valid nvidia_driver volume found for this docker container!"
    exit 1
fi

set -x

nvidia-docker run -h $HOSTNAME \
    --rm \
    --name lotus-gpu \
    --volume "$SOURCE_ROOT:$SOURCE_ROOT" \
    lotus-ubuntu16.04-cuda9.0-cudnn7.0 \
    /bin/bash $SCRIPT_DIR/run_build.sh \
    $BUILD_CONFIG &

exit $?

set -e -o

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
SOURCE_ROOT=$SCRIPT_DIR/../../../../

BUILD_CONFIG=$1

cd $SCRIPT_DIR/ubuntu16.04-cuda9.0-cudnn7.0

docker build -t lotus-ubuntu16.04-cuda9.0-cudnn7.0 .

STUB_HOME="$HOME/stub-home"

NVIDIA_DRIVER_NAME="$(docker volume ls --format "{{.Name}}" | grep -Po 'nvidia_driver_[\d.]+' | sort -V -r| head -n 1)"

if [[ -z $NVIDIA_DRIVER_NAME ]]; then
    echo "No valid nvidia_driver volume found for this docker container!"
    exit 1
fi

set -x

docker run --rm \
    --name lotus-gpu \
    --device=/dev/nvidiactl \
    --device=/dev/nvidia-uvm \
    --device=/dev/nvidia0 \
    --volume="${NVIDIA_DRIVER_NAME}:/usr/local/nvidia:ro" \
    --volume "$STUB_HOME:$HOME" \
    --volume "$SOURCE_ROOT:$SOURCE_ROOT" \
    --volume /$HOME:$HOME/workspace \
    --workdir "$HOME/workspace" \
    -e HOME=$HOME lotus-ubuntu16.04-cuda9.0-cudnn7.0 \
    /bin/bash $SCRIPT_DIR/run_build.sh \
    $BUILD_CONFIG &

exit $?
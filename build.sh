#!/bin/bash

# Other options: Debug, Release, RelWithDebInfo, MinSizeRel
TARGET=Debug
UPDATE=false
BUILD=false
BUILD_ARGS=""

for i in "$@"
do
case $i in
    -t=*|--target=*)
    TARGET="${i#*=}"
    shift # past argument=value
    ;;
    -b|--build)
    BUILD=true
    shift # past flag
    ;;
    -u|--update)
    UPDATE=true
    shift # past argument=value
    ;;
    --parallel)
    BUILD_ARGS="$BUILD_ARGS -j4"
    shift # past argument with no value
    ;;
    -h|--help)
    echo "Usage: $0 [-t|--target {Debug|Release|RelWeithDebInfo|MinSizeRel}] [-b|--build] [-u|--update] --parallel"
    echo "Default is update and build Debug"
    echo "Specify -b or -u to just build or update"
    echo "Specify --parallel for a parallel build"    
    exit 0
    ;;
    *)
          # unknown option
    ;;
esac
done

if [ "$BUILD" = false -a "$UPDATE" = false ]; then
    echo "Defaulting to updating makefiles and building"
    BUILD=true
    UPDATE=true
fi

BLD_DIR=../build/Linux/$TARGET

cd cmake

if [ "$UPDATE" = true ]; then
    echo "Updating makefiles. Target=$TARGET"
    cmake -H. -B$BLD_DIR -DCMAKE_BUILD_TYPE=$TARGET
fi

if [ "$BUILD" = true ]; then
    cmake --build $BLD_DIR -- $BUILD_ARGS
fi
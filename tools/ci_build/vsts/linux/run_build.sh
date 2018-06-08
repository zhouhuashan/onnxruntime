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

PYTHON_VERSION="py35"
source /usr/local/miniconda3/bin/activate lotus-$PYTHON_VERSION
if [ $? -ne 0 ]; then
    /usr/local/miniconda3/bin/conda env create \
        --file $SCRIPT_DIR/Conda/conda-linux-lotus-$PYTHON_VERSION-environment.yml \
        --name lotus-$PYTHON_VERSION \
        --quiet \
        --force
    source /usr/local/miniconda3/bin/activate lotus-$PYTHON_VERSION
fi

if [ $BUILD_DEVICE = "gpu" ]; then
    python $SCRIPT_DIR/../../build.py --build_dir /home/lotusdev \
        --config $BUILD_CONFIG \
        --skip_submodule_sync \
        --enable_pybind \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/local/cudnn-7.0/cuda $BUILD_EXTR_PAR
else
    python $SCRIPT_DIR/../../build.py --build_dir /home/lotusdev \
        --config $BUILD_CONFIG \
        --skip_submodule_sync \
        --enable_pybind $BUILD_EXTR_PAR
fi

source /usr/local/miniconda3/bin/deactivate lotus-$PYTHON_VERSION

# Tag: nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# Created: 2017-11-21T06:34:14.675603521Z
# Label: com.nvidia.build.id: 41212533
# Label: com.nvidia.build.ref: e0edb5359ecb7bd3d86f0c9bfa18c2260b741ebb
# Label: com.nvidia.cuda.version: 9.0.176
# Label: com.nvidia.cudnn.version: 7.0.4.31
# Label: com.nvidia.nccl.version: 2.1.2
# Ubuntu 16.04.5
FROM nvidia/cuda@sha256:33add9c50ab76b8f3a92187c0418ed600d5bea27690fda40711122fdc28ce2f4


RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        build-essential \
        git ca-certificates \
        ca-certificates \
        pkg-config \
        wget \
        zlib1g \
        zlib1g-dev \
        libssl-dev \
        curl \
        autoconf \
        sudo \
        python3-dev \
        libopenblas-dev \
        # OpenBLAS
        gfortran \
        && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Install CMake
RUN cd /usr/local/src && \ 
    wget https://cmake.org/files/v3.11/cmake-3.11.1.tar.gz && \
    tar xvf cmake-3.11.1.tar.gz && \
    cd cmake-3.11.1 && \
    ./bootstrap && \
    make && \
    make install && \
    cd .. && \
    rm -rf cmake*

# Allow configure to pick up GDK and CuDNN where it expects it.
# (Note: $CUDNN_VERSION is defined by NVidia's base image)
RUN _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2) && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/include && \
    ln -s /usr/include/cudnn.h /usr/local/cudnn-$_CUDNN_VERSION/cuda/include/cudnn.h && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64 && \
    ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64/libcudnn.so && \
    ln -s /usr/local/cudnn{-$_CUDNN_VERSION,}

ENV LD_LIBRARY_PATH /usr/local/openblas/lib:$LD_LIBRARY_PATH

ARG MINICONDA_PREFIX=/usr/local/miniconda3
ARG MINICONDA_VERSION=4.5.1
RUN MINICONDA=Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh && \
    wget --no-check-certificate --no-verbose https://repo.continuum.io/miniconda/$MINICONDA && \
    chmod a+x $MINICONDA && \
    ./$MINICONDA -b -p $MINICONDA_PREFIX && \
    rm ./$MINICONDA && \
    $MINICONDA_PREFIX/bin/conda create -c defaults -c conda-forge --quiet --yes --name lotus-py35 python=3.5.2 numpy=1.11.3 && \
    $MINICONDA_PREFIX/bin/conda clean --yes --all

ARG BUILD_UID=1000
ARG BUILD_USER=lotusdev
RUN adduser --gecos 'LOTUS Build User' --disabled-password $BUILD_USER --uid $BUILD_UID
USER $BUILD_USER
WORKDIR /home/$BUILD_USER


#!/bin/bash
set -e
DEBIAN_FRONTEND=noninteractive
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
        gfortran \
        python3-dev \
        libopenblas-dev \
        liblttng-ust0 \
        libcurl3 \
        libssl1.0.0 \
        libkrb5-3 \
        libicu55 \
        aria2 \
        bzip2 \
        unzip \
        zip \
        rsync libunwind8 \
        python3-setuptools python3-numpy python3-wheel python python3-pip

rm -rf /var/lib/apt/lists/*

mkdir -p /tmp/azcopy
aria2c -q -d /tmp/azcopy -o azcopy.tar.gz https://aka.ms/downloadazcopylinux64
tar -xf /tmp/azcopy/azcopy.tar.gz -C /tmp/azcopy
/tmp/azcopy/install.sh
rm -rf /tmp/azcopy


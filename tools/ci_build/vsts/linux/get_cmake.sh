#!/bin/bash
# copied and adapted from https://github.com/Microsoft/vsts-agent-docker/blob/master/ubuntu/16.04/standard/Dockerfile
echo Downloading cmake... \
 && curl -sL https://cmake.org/files/v3.11/cmake-3.11.1-Linux-x86_64.sh -o cmake.sh \
 && chmod +x cmake.sh \
 && ./cmake.sh --prefix=/usr/local --exclude-subdir \
 && rm -rf cmake.sh

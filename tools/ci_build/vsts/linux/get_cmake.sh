# copied from https://github.com/Microsoft/vsts-agent-docker/blob/master/ubuntu/16.04/standard/Dockerfile
echo Downloading cmake... \
 && curl -sL https://cmake.org/files/v3.9/cmake-3.9.1-Linux-x86_64.sh -o cmake.sh \
 && chmod +x cmake.sh \
 && ./cmake.sh --prefix=/usr/local --exclude-subdir \
 && rm -rf cmake.sh

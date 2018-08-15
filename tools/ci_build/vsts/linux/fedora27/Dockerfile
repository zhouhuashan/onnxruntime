FROM fedora:27
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.2"

ADD cudnn /cudnn
RUN dnf install -y curl && curl -O -L 'http://developer.download.nvidia.com/compute/cuda/repos/fedora27/x86_64/cuda-repo-fedora27-9.2.148-1.x86_64.rpm' && dnf install -y cuda-repo-fedora27-9.2.148-1.x86_64.rpm && dnf install -y cuda-9-2.x86_64 gcc cmake make ninja-build git python3-devel python3-numpy.x86_64 && cp -r /cudnn/include/* /usr/local/cuda/include/ && cp -r /cudnn/lib64/* /usr/local/cuda/lib64/ && ldconfig -v /usr/local/cuda/lib64

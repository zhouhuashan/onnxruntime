#pragma once

#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
// Enable per-thread default stream, according to https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>
#include <cudnn.h>

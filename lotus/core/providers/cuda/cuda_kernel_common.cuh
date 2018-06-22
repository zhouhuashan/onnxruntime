#pragma once
#include <stdint.h>
#include <vector>
#include <mutex>
#include <assert.h>
#include <cuda_runtime.h>
#include "cuda_call.h"

namespace Lotus {
namespace Cuda {

// We would like to use 64-bit integer to support large matrices. However, CUDA seems to support only 32-bit integer
// For now, use int32_t to ensure that both Linux and Windows see this as 32 bit integer type.

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))  // 0 based indexing

// ---------------------------------------------------------------------------
// GridDim -- helper to choose the CUDA grid dimensions
// ---------------------------------------------------------------------------

template <class INT, class INT2>
static INT CeilDiv(INT a, INT2 b)  // ceil(a/b)
{
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);  // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 1024,  // use this many threads per block
    maxWarpsPerBlock = 32,      // use this many warps per block. This means 1024 threads for warpSize=32
  };

  // use these for launching
  //   GridDim grid(NN);
  //   kernel<<<grid.m_blocksPerGrid, grid.m_threadsPerBlock, ...>>>(...)
  int blocks_per_grid_, threads_per_block_;  // (these may in the future be extended to multi-dimensional ones)
  CUDA_LONG N_;

  GridDim(CUDA_LONG N)  // linear grid
  {
    N_ = N;
    if (N == 0)  // CUDA will fail to launch with 0 blocks
      N = 1;

    // get device information
    const auto& props = GetDeviceProps();
    CUDA_LONG numProcs = props.multiProcessorCount;
    CUDA_LONG warpSize = props.warpSize;

    // distribute warps evenly over processors
    CUDA_LONG warpsPerProc = CeilDiv(N, numProcs * warpSize);

    // if too many warps per block then reduce #warps
    // This limits the number of threads to 512.
    if (warpsPerProc > maxWarpsPerBlock) {
      CUDA_LONG overBy = CeilDiv(warpsPerProc, maxWarpsPerBlock);  // we are over by this factor
      warpsPerProc = CeilDiv(warpsPerProc, overBy);
    }

    // put it back together
    threads_per_block_ = warpsPerProc * warpSize;  // =a multiple of 32 that is as close to 1024 as makes sense given NN
    blocks_per_grid_ = CeilDiv(N, threads_per_block_);
    if (blocks_per_grid_ == 1)
      threads_per_block_ = N;  // don't launch more than necessary
    assert(blocks_per_grid_ * threads_per_block_ >= N);
  }

  static const std::vector<cudaDeviceProp>& GetCachedDeviceProps() {
    std::call_once(s_cachedDevicePropsInitFlag, [=] {
      int numDevices;
      // must wait GPU idle, otherwise cudaGetDeviceProperties might fail
      CUDA_CALL_THROW(cudaDeviceSynchronize());
      CUDA_CALL_THROW(cudaGetDeviceCount(&numDevices));
      s_cachedDeviceProps.resize(numDevices);
      for (int i = 0; i < numDevices; i++)
        CUDA_CALL_THROW(cudaGetDeviceProperties(&s_cachedDeviceProps[i], i));
    });

    return s_cachedDeviceProps;
  }

  static size_t GetCurrentDeviceId() {
    int deviceId;
    cudaGetDevice(&deviceId);
    return (size_t)deviceId;
  }

  // get device properties of current device
  static const cudaDeviceProp& GetDeviceProps() {
    const auto& cachedDevicesProps = GetCachedDeviceProps();
    return cachedDevicesProps[GetCurrentDeviceId()];
  }

  // compute our location on the grid
  static __device__ CUDA_LONG GetLinearThreadId() {
    return blockDim.x * blockIdx.x + threadIdx.x;
  }

 private:
  static std::vector<cudaDeviceProp> s_cachedDeviceProps;
  static std::once_flag s_cachedDevicePropsInitFlag;
};

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N) \
  CUDA_LONG id = GridDim::GetLinearThreadId();     \
  if (id >= N)                                     \
    return;

}  // namespace Cuda
}  // namespace Lotus

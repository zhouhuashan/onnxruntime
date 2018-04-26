#include "core/framework/allocator.h"
#include <stdlib.h>
#include <sstream>

namespace Lotus {

void* CUDAAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    cudaMalloc((void**)&p, size);
  }
  return p;
}

void CUDAAllocator::Free(void* p) {
  cudaFree(p);
}

const AllocatorInfo& CUDAAllocator::Info() const {
  static AllocatorInfo cudaAllocatorInfo(CUDA, AllocatorType::kDeviceAllocator);
  return cudaAllocatorInfo;
}

}  // namespace Lotus

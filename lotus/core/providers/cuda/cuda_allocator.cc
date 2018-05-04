#include "cuda_common.h"
#include "cuda_allocator.h"
#include "core/framework/allocatormgr.h"

namespace Lotus {

REGISTER_DEVICE_ALLOCATOR(
    Cuda,
    []() { return std::make_unique<CUDAAllocator>(); },
    std::numeric_limits<size_t>::max())  //TODO: set correct cpu memory limit?)

void* CUDAAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL(cudaMalloc((void**)&p, size));
  }
  return p;
}

void CUDAAllocator::Free(void* p) {
  CUDA_CALL(cudaFree(p));
}

const AllocatorInfo& CUDAAllocator::Info() const {
  static AllocatorInfo cudaAllocatorInfo(CUDA, AllocatorType::kDeviceAllocator);
  return cudaAllocatorInfo;
}

}  // namespace Lotus

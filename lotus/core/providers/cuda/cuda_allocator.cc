#include "cuda_common.h"
#include "cuda_allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "cuda_fence.h"

namespace Lotus {

REGISTER_DEVICE_ALLOCATOR_WITH_MEM_TYPE(
    Cuda,
    [](int id) { return std::make_unique<CUDAAllocator>(id); },
    std::numeric_limits<size_t>::max(),  //TODO: set correct gpu memory limit?
    kMemTypeDefault)

REGISTER_DEVICE_ALLOCATOR_WITH_MEM_TYPE(
    CudaPinned,
    [](int) { return std::make_unique<CUDAPinnedAllocator>(); },
    std::numeric_limits<size_t>::max(),  //TODO: set correct cpu memory limit?
    kMemTypeCPU)

static CUDAExecutionProvider* GetCUDAExecutionProvider(const SessionState* session_state) {
  return dynamic_cast<CUDAExecutionProvider*>(session_state->GetExecutionProvider(LotusIR::kCudaExecutionProvider));
}

void CUDAAllocator::CheckDevice() const {
#ifdef _DEBUG
  // check device to match at debug build
  // if it's expected to change, call cudaSetDevice instead of the check
  int current_device;
  CUDA_CALL(cudaGetDevice(&current_device));
  LOTUS_ENFORCE(current_device == device_id_);
#endif
}

void* CUDAAllocator::Alloc(size_t size) {
  CheckDevice();
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL(cudaMalloc((void**)&p, size));
  }
  return p;
}

void CUDAAllocator::Free(void* p) {
  CheckDevice();
  CUDA_CALL(cudaFree(p));
}

const AllocatorInfo& CUDAAllocator::Info() const {
  static AllocatorInfo cudaAllocatorInfo(CUDA, AllocatorType::kDeviceAllocator, device_id_, kMemTypeDefault);
  return cudaAllocatorInfo;
}

FencePtr CUDAAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CUDAFence>(GetCUDAExecutionProvider(session_state));
}

void* CUDAPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL(cudaMallocHost((void**)&p, size));
  }
  return p;
}

void CUDAPinnedAllocator::Free(void* p) {
  CUDA_CALL(cudaFreeHost(p));
}

const AllocatorInfo& CUDAPinnedAllocator::Info() const {
  static AllocatorInfo cudaAllocatorInfo(CUDA_PINNED, AllocatorType::kDeviceAllocator, 0, kMemTypeCPU);
  return cudaAllocatorInfo;
}

FencePtr CUDAPinnedAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CUDAFence>(GetCUDAExecutionProvider(session_state));
}

}  // namespace Lotus

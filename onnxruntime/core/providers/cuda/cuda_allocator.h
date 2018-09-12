#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {

class CUDAAllocator : public IDeviceAllocator {
 public:
  CUDAAllocator(int device_id) : device_id_(device_id), info_(CUDA, AllocatorType::kDeviceAllocator, device_id, kMemTypeDefault) {}
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const AllocatorInfo& Info() const override;
  virtual FencePtr CreateFence(const SessionState* session_state) override;

 private:
  void CheckDevice() const;

 private:
  const int device_id_;
  const AllocatorInfo info_;
};

class CUDAPinnedAllocator : public IDeviceAllocator {
 public:
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const AllocatorInfo& Info() const override;
  virtual FencePtr CreateFence(const SessionState* session_state) override;
};

}  // namespace onnxruntime

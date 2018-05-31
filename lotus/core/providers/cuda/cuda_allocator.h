#pragma once

#include "core/framework/allocator.h"

namespace Lotus {

class CUDAAllocator : public IDeviceAllocator {
 public:
  CUDAAllocator(int device_id) : device_id_(device_id) {}
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const AllocatorInfo& Info() const override;
  virtual FencePtr CreateFence(const SessionState* session_state) override;

 private:
  void CheckDevice() const;

 private:
  int device_id_;
};

class CUDAPinnedAllocator : public IDeviceAllocator {
 public:
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const AllocatorInfo& Info() const override;
  virtual FencePtr CreateFence(const SessionState* session_state) override;
};

}  // namespace Lotus

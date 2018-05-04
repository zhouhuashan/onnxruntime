#pragma once

#include "core/framework/allocator.h"

namespace Lotus {

class CUDAAllocator : public IDeviceAllocator {
 public:
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const AllocatorInfo& Info() const override;
};

}  // namespace Lotus

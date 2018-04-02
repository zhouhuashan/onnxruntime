#pragma once

#include <string>
#include "core/framework/allocator.h"

namespace Lotus {
// The interface for arena which manage memory allocations
// Arena will hold a pool of pre-allocate memories and manage their lifecycle.
// Need an underline IResourceAllocator to allocate memories.
// The setting like max_chunk_size is init by IDeviceDescriptor from resource allocator
class IArenaAllocator : public IAllocator {
 public:
  virtual ~IArenaAllocator() {}
  // Alloc call need to be thread safe.
  virtual void* Alloc(size_t size) = 0;
  // Free call need to be thread safe.
  virtual void Free(void* p) = 0;
  virtual size_t Used() const = 0;
  virtual size_t Max() const = 0;
  virtual const AllocatorInfo& Info() const = 0;
  // allocate host pinned memory?
};

// Dummy Arena which just call underline device allocator directly.
class DummyArena : public IArenaAllocator {
 public:
  DummyArena(IDeviceAllocator* resource_allocator)
      : allocator_(resource_allocator),
        info_(resource_allocator->Info().name, AllocatorType::kArenaAllocator, resource_allocator->Info().id) {
  }

  virtual ~DummyArena() {}

  virtual void* Alloc(size_t size) override {
    if (size == 0)
      return nullptr;
    return allocator_->Alloc(size);
  }

  virtual void Free(void* p) override {
    allocator_->Free(p);
  }

  virtual size_t Used() const override {
    LOTUS_NOT_IMPLEMENTED;
  }

  virtual size_t Max() const override {
    LOTUS_NOT_IMPLEMENTED;
  }

  virtual const AllocatorInfo& Info() const override {
    return info_;
  }

 private:
  IDeviceAllocator* allocator_;
  AllocatorInfo info_;
};
}  // namespace Lotus

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
  // The chunck allocated by Reserve call won't be reused with other request.
  // It will be return to the devices when it is freed.
  // Reserve call need to be thread safe.
  virtual void* Reserve(size_t size) = 0;
  // Free call need to be thread safe.
  virtual void Free(void* p) = 0;
  virtual size_t Used() const = 0;
  virtual size_t Max() const = 0;
  virtual const AllocatorInfo& Info() const = 0;
  // allocate host pinned memory?
};

using ArenaPtr = std::shared_ptr<IArenaAllocator>;

// Dummy Arena which just call underline device allocator directly.
class DummyArena : public IArenaAllocator {
 public:
  DummyArena(std::unique_ptr<IDeviceAllocator> resource_allocator)
      : allocator_(std::move(resource_allocator)),
        info_(allocator_->Info().name, AllocatorType::kArenaAllocator, allocator_->Info().id) {
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

  virtual void* Reserve(size_t size) override {
    return Alloc(size);
  }

  virtual size_t Used() const override {
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  virtual size_t Max() const override {
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  virtual const AllocatorInfo& Info() const override {
    return info_;
  }

 private:
  std::unique_ptr<IDeviceAllocator> allocator_;
  AllocatorInfo info_;
};
}  // namespace Lotus

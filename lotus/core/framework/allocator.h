#pragma once

#include <map>
#include <string>
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/common/status.h"

namespace Lotus {
#define CPU "Cpu"
#define CUDA "Cuda"
#define CUDA_PINNED "CudaPinned"

enum AllocatorType {
  kDeviceAllocator = 0,
  kArenaAllocator = 1
};

// memory types for allocator, exec provider specific types should be extended in each provider
enum MemType : int {
  kMemTypeCPU = -1,     // CPU accessible memory managed by non-CPU execution provider
  kMemTypeDefault = 0,  // the default allocator for execution provider
};

struct AllocatorInfo {
  // use string for name, so we could have customized allocator in execution provider.
  std::string name;
  int id;
  MemType mem_type;
  AllocatorType type;

 public:
  AllocatorInfo(const std::string& name, AllocatorType type, int id1 = 0, MemType mem_type1 = kMemTypeDefault)
      : name(name),
        id(id1),
        mem_type(mem_type1),
        type(type) {}

  inline bool operator==(const AllocatorInfo& other) const {
    return name == other.name && mem_type == other.mem_type && type == other.type && id == other.id;
  }

  // To make AllocatorInfo become a valid key in std map
  inline bool operator<(const AllocatorInfo& other) const {
    if (type != other.type)
      return type < other.type;
    else if (mem_type != other.mem_type)
      return mem_type < other.mem_type;
    else if (id != other.id)
      return id < other.id;
    else
      return name < other.name;
  }
};

class IAllocator {
 public:
  virtual ~IAllocator() {}
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
  virtual const AllocatorInfo& Info() const = 0;
};

// The resource allocator on a physical device.
// This allocator will directly allocate resource from system call
class IDeviceAllocator : public IAllocator {
 public:
  virtual ~IDeviceAllocator() {}
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
  virtual const AllocatorInfo& Info() const = 0;
  virtual bool AllowsArena() const { return true; }
};

class CPUAllocator : public IDeviceAllocator {
 public:
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const AllocatorInfo& Info() const override;
};

using AllocatorPtr = std::shared_ptr<IAllocator>;

}  // namespace Lotus

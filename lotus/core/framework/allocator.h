#pragma once

#include <map>
#include <string>
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/common/status.h"

namespace Lotus {
#define CPU "CPU"
#define CUDA "CUDA"

enum AllocatorType {
  kDeviceAllocator = 0,
  kArenaAllocator = 1
};

struct AllocatorInfo {
  // use string for name, so we could have customized allocator in execution provider.
  std::string name;
  int id;
  AllocatorType type;

 public:
  AllocatorInfo(const std::string& name, AllocatorType type, const int id = 0)
      : name(name),
        id(id),
        type(type) {}

  inline bool operator==(const AllocatorInfo& other) const {
    return name == other.name && id == other.id && type == other.type;
  }

  // To make AllocatorInfo become a valid key in std map
  inline bool operator<(const AllocatorInfo& other) const {
    if (type != other.type)
      return type < other.type;
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
};

class CPUAllocator : public IDeviceAllocator {
 public:
  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  virtual const AllocatorInfo& Info() const override;
};

}  // namespace Lotus

#pragma once

#include <functional>
#include <map>
#include <string>
#include <type_traits>

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/common/status.h"
#include "core/framework/fence.h"

namespace Lotus {
constexpr const char* CPU = "Cpu";
constexpr const char* CUDA = "Cuda";
constexpr const char* CUDA_PINNED = "CudaPinned";
constexpr const char* MKLDNN = "MklDnn";
constexpr const char* MKLDNN_CPU = "MklDnnCpu";

// forward declaration
class SessionState;

enum AllocatorType {  // TODO use enum class
  kDeviceAllocator = 0,
  kArenaAllocator = 1
};

// memory types for allocator, exec provider specific types should be extended in each provider
enum MemType : int {                // TODO use enum class
  kMemTypeCPUInput = -2,            // Any CPU memory used by non-CPU execution provider
  kMemTypeCPUOutput = -1,           // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
  kMemTypeCPU = kMemTypeCPUOutput,  // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
  kMemTypeDefault = 0,              // the default allocator for execution provider
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
    return mem_type == other.mem_type && type == other.type && id == other.id && name == other.name;
  }

  // To make AllocatorInfo become a valid key in std map
  inline bool operator<(const AllocatorInfo& other) const {
    if (type != other.type)
      return type < other.type;
    if (mem_type != other.mem_type)
      return mem_type < other.mem_type;
    if (id != other.id)
      return id < other.id;

    return name < other.name;
  }

  inline std::string ToString() const {
    std::ostringstream ostr;
    ostr << "AllocatorInfo: ["
         << " name:" << name
         << " id:" << id
         << " mem_type:" << mem_type
         << " type:" << type
         << "]";
    return ostr.str();
  }
};

std::ostream& operator<<(std::ostream& out, const AllocatorInfo& info);

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

class IAllocator {
 public:
  virtual ~IAllocator() = default;
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
  virtual const AllocatorInfo& Info() const = 0;

  // optional CreateFence interface, as provider like DML has its own fence
  virtual FencePtr CreateFence(const SessionState* /*unused*/) { return nullptr; }

  /// Create a std::unique_ptr that is allocated and freed by the provided IAllocator.
  /// @param allocator The allocator.
  /// @param count_or_bytes The exact bytes to allocate if T is void, otherwise the number of elements to allocate.
  /// @returns std::unique_ptr with allocated memory and deleter.
  template <typename T>
  static IAllocatorUniquePtr<T> MakeUniquePtr(std::shared_ptr<IAllocator> allocator, size_t count_or_bytes) {
    // for now limit to fundamental types. we could support others, but to do so either we or the caller
    // needs to call the dtor for the objects, for buffers allocated on device we don't have destructor
    //static_assert(std::is_fundamental<T>::value, "Fundamental type required as no destructors are called.");

    size_t alloc_size = count_or_bytes;

    // if T is not void, 'count_or_bytes' == number of items so allow for that
    if (!std::is_void<T>::value)
      // sizeof(void) isn't valid, but the compiler isn't smart enough to ignore that this line isn't
      // reachable if T is void. use std::conditional to 'use' void* in the sizeof call
      alloc_size *= sizeof(typename std::conditional<std::is_void<T>::value, void*, T>::type);

    return IAllocatorUniquePtr<T>{
        static_cast<T*>(allocator->Alloc(alloc_size)),  // allocate
        [=](T* ptr) { allocator->Free(ptr); }};         // capture IAllocator so it's always valid, and use as deleter
  }
};

// The resource allocator on a physical device.
// This allocator will directly allocate resource from system call
class IDeviceAllocator : public IAllocator {
 public:
  ~IDeviceAllocator() override = default;
  void* Alloc(size_t size) override = 0;
  void Free(void* p) override = 0;
  const AllocatorInfo& Info() const override = 0;
  virtual bool AllowsArena() const { return true; }
};

class CPUAllocator : public IDeviceAllocator {
 public:
  void* Alloc(size_t size) override;
  void Free(void* p) override;
  const AllocatorInfo& Info() const override;
};

using AllocatorPtr = std::shared_ptr<IAllocator>;

}  // namespace Lotus

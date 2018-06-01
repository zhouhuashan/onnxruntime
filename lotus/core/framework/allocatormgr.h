#pragma once

#include "core/framework/arena.h"

namespace Lotus {

using DeviceAllocatorFactory = std::function<std::unique_ptr<IDeviceAllocator> (int)>;

struct DeviceAllocatorRegistrationInfo {
  MemType mem_type;
  DeviceAllocatorFactory factory;
  size_t max_mem;
};

AllocatorPtr CreateAllocator(DeviceAllocatorRegistrationInfo info, int device_id = 0);

class DeviceAllocatorRegistry {
 public:
  void RegisterDeviceAllocator(std::string&& name, DeviceAllocatorFactory factory, size_t max_mem, MemType mem_type = kMemTypeDefault) {
    DeviceAllocatorRegistrationInfo info({mem_type, factory, max_mem});
    device_allocator_registrations_.insert(std::make_pair(std::move(name), std::move(info)));
  }

  const std::map<std::string, DeviceAllocatorRegistrationInfo>& AllRegistrations() const {
    return device_allocator_registrations_;
  }

  static DeviceAllocatorRegistry& Instance();

 private:
  std::map<std::string, DeviceAllocatorRegistrationInfo> device_allocator_registrations_;
};

#define REGISTER_DEVICE_ALLOCATOR_WITH_MEM_TYPE(name, func, max_mem, mem_type) \
  class REGISTER_DEVICE_ALLOCATOR_##name {                                     \
   public:                                                                     \
    REGISTER_DEVICE_ALLOCATOR_##name() {                                       \
      DeviceAllocatorRegistry::Instance().RegisterDeviceAllocator(             \
          std::string(#name), func, max_mem, mem_type);                        \
    }                                                                          \
  };                                                                           \
  REGISTER_DEVICE_ALLOCATOR_##name g_REGISTER_DEVICE_ALLOCATOR_##name;

#define REGISTER_DEVICE_ALLOCATOR(name, func, max_mem) \
  REGISTER_DEVICE_ALLOCATOR_WITH_MEM_TYPE(name, func, max_mem, kMemTypeDefault)

}  // namespace Lotus

#pragma once

#include "core/framework/arena.h"

namespace Lotus {
struct DeviceAllocatorRegistrationInfo {
  std::function<std::unique_ptr<IDeviceAllocator>()> factory;
  size_t max_mem;
};

ArenaPtr CreateArena(DeviceAllocatorRegistrationInfo info);

class DeviceAllocatorRegistry {
 public:
  void RegisterDeviceAllocator(std::string&& name, std::function<std::unique_ptr<IDeviceAllocator>()> factory, size_t max_mem) {
    DeviceAllocatorRegistrationInfo info({factory, max_mem});
    device_allocator_registrations_.insert(std::make_pair(std::move(name), std::move(info)));
  }

  const std::map<std::string, DeviceAllocatorRegistrationInfo>& AllRegistrations() const {
    return device_allocator_registrations_;
  }

  static DeviceAllocatorRegistry& Instance();

 private:
  std::map<std::string, DeviceAllocatorRegistrationInfo> device_allocator_registrations_;
};

#define REGISTER_DEVICE_ALLOCATOR(name, func, max_mem)                                                \
  class REGISTER_DEVICE_ALLOCATOR_##name {                                                            \
   public:                                                                                            \
    REGISTER_DEVICE_ALLOCATOR_##name() {                                                              \
      DeviceAllocatorRegistry::Instance().RegisterDeviceAllocator(std::string(#name), func, max_mem); \
    }                                                                                                 \
  };                                                                                                  \
  REGISTER_DEVICE_ALLOCATOR_##name g_REGISTER_DEVICE_ALLOCATOR_##name;

}  // namespace Lotus

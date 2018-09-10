#pragma once

#include "core/common/common.h"
#include "core/framework/arena.h"

namespace onnxruntime {

using DeviceAllocatorFactory = std::function<std::unique_ptr<IDeviceAllocator>(int)>;

struct DeviceAllocatorRegistrationInfo {
  MemType mem_type;
  DeviceAllocatorFactory factory;
  size_t max_mem;
};

AllocatorPtr CreateAllocator(DeviceAllocatorRegistrationInfo info, int device_id = 0);

class DeviceAllocatorRegistry {
 public:
  void RegisterDeviceAllocator(std::string&& name, DeviceAllocatorFactory factory, size_t max_mem,
                               MemType mem_type = kMemTypeDefault) {
    DeviceAllocatorRegistrationInfo info({mem_type, factory, max_mem});
    device_allocator_registrations_.emplace(std::move(name), std::move(info));
  }

  const std::map<std::string, DeviceAllocatorRegistrationInfo>& AllRegistrations() const {
    return device_allocator_registrations_;
  }

  static DeviceAllocatorRegistry& Instance();

 private:
  DeviceAllocatorRegistry() = default;
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(DeviceAllocatorRegistry);

  std::map<std::string, DeviceAllocatorRegistrationInfo> device_allocator_registrations_;
};

}  // namespace onnxruntime

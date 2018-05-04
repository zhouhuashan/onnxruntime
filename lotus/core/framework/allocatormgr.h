#pragma once

#include "core/framework/arena.h"

namespace Lotus {
class AllocatorManager {
 public:
  // the allocator manager is a global object for entire Process.
  // all the inference engine in the same Process will use the same allocator manager.
  static AllocatorManager& Instance();

  /**
        Create an AllocatorManager instance. This is expected to be called once and remain valid
        for the duration of execution. It will populate Instance() for convenient access.
        */
  static Status Create(std::unique_ptr<AllocatorManager>& allocator_manager);

  /**
        Destruct th AllocatorManager. Will unset Instance().
        */
  ~AllocatorManager();

  IArenaAllocator& GetArena(const std::string& name, const int id = 0);

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(AllocatorManager);

  AllocatorManager();
  Status InitializeAllocators();

  std::unordered_map<std::string, std::unique_ptr<IArenaAllocator>> arena_map_;
  bool owns_instance_;
};

class DeviceAllocatorRegistry {
 public:
  struct DeviceAllocatorRegistrationInfo {
    std::function<std::unique_ptr<IDeviceAllocator>()> factory;
    size_t max_mem;
  };

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

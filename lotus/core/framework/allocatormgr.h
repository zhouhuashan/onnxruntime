#pragma once

#include "core/framework/arena.h"

namespace Lotus {
class AllocatorManager {
  friend class Initializer;

 public:
  // the allocator manager is a global object for entire Process.
  // all the inference engine in the same Process will use the same allocator manager.
  // TODO(Task:151) AllocatorManager::Instance could return reference
  static AllocatorManager& Instance() {
    static AllocatorManager manager;
    return manager;
  }

  IArenaAllocator& GetArena(const std::string& name, const int id = 0);

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(AllocatorManager);
  AllocatorManager() = default;

  // after add allocator, allocator manager will take the ownership.
  Status AddDeviceAllocator(IDeviceAllocator* allocator, const bool create_arena = true);
  Status AddArenaAllocator(IArenaAllocator* allocator);

  static std::string GetAllocatorId(const std::string& name, const int id, const bool isArena);

  Status InitializeAllocators();
};
}  // namespace Lotus

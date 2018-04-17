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
}  // namespace Lotus

#pragma once
#include "core/framework/arena.h"
namespace Lotus {
namespace Test {
class AllocatorManager {
 public:
  // the allocator manager is a just for onnx runner to allocate space for input/output tensors.
  // Lotus session will use the allocator owned by execution provider.
  static AllocatorManager& Instance();

  /**
  Destruct th AllocatorManager. Will unset Instance().
  */
  ~AllocatorManager();

  AllocatorPtr GetAllocator(const std::string& name, const int id = 0, bool arena = true);

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(AllocatorManager);

  AllocatorManager();
  Status InitializeAllocators();

  std::unordered_map<std::string, AllocatorPtr> map_;
  bool owns_instance_;
};
}  // namespace Test
}  // namespace Lotus

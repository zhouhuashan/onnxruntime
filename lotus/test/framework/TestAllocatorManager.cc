#include "test/framework/TestAllocatorManager.h"
#include "core/framework/allocatormgr.h"
namespace Lotus {
namespace Test {

static std::string GetAllocatorId(const std::string& name, const int id, const bool isArena) {
  std::stringstream ss;
  if (isArena)
    ss << "arena_";
  else
    ss << "device_";
  ss << name << "_" << id;
  return ss.str();
}

static Status RegisterArena(std::unordered_map<std::string, ArenaPtr>& arena_map,
                            std::unique_ptr<IDeviceAllocator> allocator, size_t /*memory_limit*/) {
  auto& info = allocator->Info();
  auto allocator_id = GetAllocatorId(info.name, info.id, true);
  auto arena_id = GetAllocatorId(info.name, info.id, true);

  auto status = Status::OK();
  if (arena_map.find(arena_id) != arena_map.end())
    status = Status(Common::LOTUS, StatusCode::FAIL, "arena already exists");
  else {
    arena_map[arena_id] = std::make_shared<DummyArena>(std::move(allocator));
  }

  return status;
}

AllocatorManager& AllocatorManager::Instance() {
  static AllocatorManager s_instance_;
  return s_instance_;
}

AllocatorManager::AllocatorManager() {
  InitializeAllocators();
}

Status AllocatorManager::InitializeAllocators() {
  Status status = Status::OK();

  for (const auto& pair : DeviceAllocatorRegistry::Instance().AllRegistrations()) {
    if (status.IsOK()) {
      auto allocator = std::unique_ptr<IDeviceAllocator>(pair.second.factory());
      status = RegisterArena(arena_map_, std::move(allocator), pair.second.max_mem);
    }
  }
  return status;
}

AllocatorManager::~AllocatorManager() {
}

ArenaPtr AllocatorManager::GetArena(const std::string& name, const int id) {
  auto arena_id = GetAllocatorId(name, id, true);
  auto entry = arena_map_.find(arena_id);
  LOTUS_ENFORCE(entry != arena_map_.end(), "Arena not found:", arena_id);
  return entry->second;
}
}  // namespace Test
}  // namespace Lotus

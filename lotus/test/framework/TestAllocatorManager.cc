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

static Status RegisterAllocator(std::unordered_map<std::string, AllocatorPtr>& map,
                                std::unique_ptr<IDeviceAllocator> allocator, size_t /*memory_limit*/,
                                bool use_arena) {
  auto& info = allocator->Info();
  auto allocator_id = GetAllocatorId(info.name, info.id, use_arena);

  auto status = Status::OK();
  if (map.find(allocator_id) != map.end())
    status = Status(Common::LOTUS, Common::FAIL, "allocator already exists");
  else {
    if (use_arena)
      map[allocator_id] = std::make_shared<DummyArena>(std::move(allocator));
    else
      map[allocator_id] = std::move(allocator);
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
      auto allocator = std::unique_ptr<IDeviceAllocator>(pair.second.factory(0));
      bool use_arena = allocator->AllowsArena();
      status = RegisterAllocator(map_, std::move(allocator), pair.second.max_mem, use_arena);
    }
  }
  return status;
}

AllocatorManager::~AllocatorManager() {
}

AllocatorPtr AllocatorManager::GetAllocator(const std::string& name, const int id, bool arena) {
  auto allocator_id = GetAllocatorId(name, id, arena);
  auto entry = map_.find(allocator_id);
  LOTUS_ENFORCE(entry != map_.end(), "Allocator not found:", allocator_id);
  return entry->second;
}
}  // namespace Test
}  // namespace Lotus

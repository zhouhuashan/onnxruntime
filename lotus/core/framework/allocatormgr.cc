#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <limits>

namespace Lotus {

using namespace Lotus::Common;

static std::mutex& Mutex() {
  static std::mutex mutex;
  return mutex;
}

static std::string GetAllocatorId(const std::string& name, const int id, const bool isArena) {
  std::stringstream ss;
  if (isArena)
    ss << "arena_";
  else
    ss << "device_";
  ss << name << "_" << id;
  return ss.str();
}

static Status RegisterBFCArena(std::unordered_map<std::string, std::unique_ptr<IArenaAllocator>>& arena_map,
                               std::unique_ptr<IDeviceAllocator> allocator, size_t memory_limit) {
  auto& info = allocator->Info();
  auto allocator_id = GetAllocatorId(info.name, info.id, true);
  auto arena_id = GetAllocatorId(info.name, info.id, true);

  auto status = Status::OK();
  if (arena_map.find(arena_id) != arena_map.end())
    status = Status(LOTUS, StatusCode::FAIL, "arena already exists");
  else {
    arena_map[arena_id] = std::unique_ptr<IArenaAllocator>(
        std::make_unique<BFCArena>(std::move(allocator), memory_limit));
  }

  return status;
}

static AllocatorManager* s_instance_ = nullptr;

Status AllocatorManager::Create(std::unique_ptr<AllocatorManager>& allocator_manager) {
  std::lock_guard<std::mutex> lock(Mutex());

  // private ctor so can't use make_unique
  allocator_manager.reset(new AllocatorManager());
  auto status = allocator_manager->InitializeAllocators();

  if (status == Status::OK()) {
    s_instance_ = allocator_manager.get();
    allocator_manager->owns_instance_ = true;
  }

  return status;
}

AllocatorManager::AllocatorManager() {
  LOTUS_ENFORCE(s_instance_ == nullptr, "AllocatorManager instance already exists.");
}

AllocatorManager& AllocatorManager::Instance() {
  LOTUS_ENFORCE(s_instance_ != nullptr, "Create a Lotus::Environment instance before executing the model.");
  return *s_instance_;
}

Status AllocatorManager::InitializeAllocators() {
  Status status = Status::OK();

  for (const auto& pair : DeviceAllocatorRegistry::Instance().AllRegistrations()) {
    if (status.IsOK()) {
      auto allocator = std::unique_ptr<IDeviceAllocator>(pair.second.factory());
      status = RegisterBFCArena(arena_map_, std::move(allocator), pair.second.max_mem);
      LOGS_DEFAULT(ERROR) << "Failed to create BFCArena for " << pair.first;
    }
  }

  return status;
}

AllocatorManager::~AllocatorManager() {
  std::lock_guard<std::mutex> lock(Mutex());

  // if we set s_instance_ we need to unset it
  if (owns_instance_)
    s_instance_ = nullptr;

  for (auto& entry : arena_map_) {
    LOGS_DEFAULT(INFO) << "Freeing arena: " << entry.first;
    entry.second.reset();
  }
}

IArenaAllocator& AllocatorManager::GetArena(const std::string& name, const int id) {
  auto arena_id = GetAllocatorId(name, id, true);

  std::lock_guard<std::mutex> lock(Mutex());
  auto entry = arena_map_.find(arena_id);
  LOTUS_ENFORCE(entry != arena_map_.end(), "Arena not found:", arena_id);

  return *(entry->second);
}

DeviceAllocatorRegistry& DeviceAllocatorRegistry::Instance() {
  static DeviceAllocatorRegistry s_instance;
  return s_instance;
}

}  // namespace Lotus

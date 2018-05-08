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

static Status RegisterAllocator(std::unordered_map<std::string, std::unique_ptr<IAllocator>>& alloc_map,
                                std::unique_ptr<IDeviceAllocator> allocator, size_t memory_limit,
                                bool use_arena) {
  auto& info = allocator->Info();
  auto alloc_id = GetAllocatorId(info.name, info.id, use_arena);

  auto status = Status::OK();
  if (alloc_map.find(alloc_id) != alloc_map.end())
    status = Status(LOTUS, StatusCode::FAIL, "Allocator already exists");
  else {
    if (use_arena) {
      alloc_map[alloc_id] = std::unique_ptr<IAllocator>(
          std::make_unique<BFCArena>(std::move(allocator), memory_limit));
    } else {
      alloc_map[alloc_id] = std::move(allocator);
    }
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
      const auto& name = pair.first;
      const auto& info = pair.second;
      for (bool use_arena : std::vector<bool>({true, false})) {
        status = RegisterAllocator(alloc_map_, std::unique_ptr<IDeviceAllocator>(info.factory()), info.max_mem, use_arena);
        if (!status.IsOK())
          LOGS_DEFAULT(ERROR) << "Failed to create " << (use_arena ? "arena" : "") << " for " << name;
      }
    }
  }
  return status;
}

AllocatorManager::~AllocatorManager() {
  std::lock_guard<std::mutex> lock(Mutex());

  // if we set s_instance_ we need to unset it
  if (owns_instance_)
    s_instance_ = nullptr;

  for (auto& entry : alloc_map_) {
    LOGS_DEFAULT(INFO) << "Freeing arena: " << entry.first;
    entry.second.reset();
  }
}

IAllocator& AllocatorManager::GetAllocator(const std::string& name, const int id, bool use_arena) {
  auto alloc_id = GetAllocatorId(name, id, use_arena);

  std::lock_guard<std::mutex> lock(Mutex());
  auto entry = alloc_map_.find(alloc_id);
  LOTUS_ENFORCE(entry != alloc_map_.end(), "Allocator not found:", alloc_id);

  return *(entry->second);
}

IArenaAllocator& AllocatorManager::GetArena(const std::string& name, const int id) {
  return *dynamic_cast<IArenaAllocator*>(&GetAllocator(name, id, true));
}

DeviceAllocatorRegistry& DeviceAllocatorRegistry::Instance() {
  static DeviceAllocatorRegistry s_instance;
  return s_instance;
}

}  // namespace Lotus

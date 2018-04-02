#include "core/framework/allocatormgr.h"
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace Lotus {

using namespace Lotus::Common;

std::unordered_map<std::string, std::unique_ptr<IDeviceAllocator>>& _getDeviceAllocatorMap() {
  static std::unordered_map<std::string, std::unique_ptr<IDeviceAllocator>> gDeviceAllocatorMap;
  return gDeviceAllocatorMap;
}

std::unordered_map<std::string, std::unique_ptr<IArenaAllocator>>& _getArenaAllocatorMap() {
  static std::unordered_map<std::string, std::unique_ptr<IArenaAllocator>> gArenaAllocatorMap;
  return gArenaAllocatorMap;
}

std::mutex& _getLocalMutex() {
  static std::mutex gMtx;
  return gMtx;
}

Status AllocatorManager::AddDeviceAllocator(IDeviceAllocator* allocator, const bool create_arena) {
  std::lock_guard<std::mutex> lock(_getLocalMutex());
  auto& device_map = _getDeviceAllocatorMap();
  auto& arena_map = _getArenaAllocatorMap();
  auto& info = allocator->Info();
  auto allocator_id = GetAllocatorId(info.name, info.id, false);
  if (device_map.find(allocator_id) != device_map.end())
    return Status(LOTUS, FAIL, "device allocator already exist");

  auto arena_id = GetAllocatorId(info.name, info.id, true);
  if (create_arena && arena_map.find(arena_id) != arena_map.end())
    return Status(LOTUS, FAIL, "arena already exist");

  device_map[allocator_id] = std::unique_ptr<IDeviceAllocator>(allocator);
  arena_map[arena_id] = std::unique_ptr<IArenaAllocator>(new DummyArena(allocator));
  return Status::OK();
}

Status AllocatorManager::AddArenaAllocator(IArenaAllocator* allocator) {
  std::lock_guard<std::mutex> lock(_getLocalMutex());
  auto& arena_map = _getArenaAllocatorMap();
  auto& info = allocator->Info();
  auto arena_id = GetAllocatorId(info.name, info.id, true);
  if (arena_map.find(arena_id) != arena_map.end())
    return Status(LOTUS, FAIL, "arena already exist");
  arena_map[arena_id] = std::unique_ptr<IArenaAllocator>(allocator);
  return Status::OK();
}

Status AllocatorManager::InitializeAllocators() {
  //right now we only have cpu allocator;
  return AddDeviceAllocator(new CPUAllocator());
}

IArenaAllocator& AllocatorManager::GetArena(const std::string& name, const int id) {
  auto& arena_map = _getArenaAllocatorMap();
  auto arena_id = GetAllocatorId(name, id, true);

  std::lock_guard<std::mutex> lock(_getLocalMutex());
  if (arena_map.find(arena_id) == arena_map.end())
    throw std::logic_error("Arena not found.");
  return *(arena_map[arena_id]);
}

std::string AllocatorManager::GetAllocatorId(const std::string& name, const int id, const bool isArena) {
  std::stringstream ss;
  if (isArena)
    ss << "arena_";
  else
    ss << "device_";
  ss << name << "_" << id;
  return ss.str();
}
}  // namespace Lotus

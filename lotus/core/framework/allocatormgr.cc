#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <limits>

namespace Lotus {

using namespace Lotus::Common;

std::unordered_map<std::string, std::unique_ptr<IArenaAllocator>>& _getArenaAllocatorMap() {
  static std::unordered_map<std::string, std::unique_ptr<IArenaAllocator>> gArenaAllocatorMap;
  return gArenaAllocatorMap;
}

std::mutex& _getLocalMutex() {
  static std::mutex gMtx;
  return gMtx;
}

Status AllocatorManager::RegisterBFCArena(IDeviceAllocator* allocator, size_t memory_limit) {
  std::lock_guard<std::mutex> lock(_getLocalMutex());
  auto& arena_map = _getArenaAllocatorMap();
  auto& info = allocator->Info();
  auto allocator_id = GetAllocatorId(info.name, info.id, true);
  auto arena_id = GetAllocatorId(info.name, info.id, true);
  if (arena_map.find(arena_id) != arena_map.end())
    return Status(LOTUS, FAIL, "arena already exist");

  arena_map[arena_id] = std::unique_ptr<IArenaAllocator>(new BFCArena(allocator, memory_limit));
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
  //TODO: set correct cpu memory limit?
  static const size_t cpu_memory_limit = std::numeric_limits<size_t>::max();
  return RegisterBFCArena(new CPUAllocator(), cpu_memory_limit);
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

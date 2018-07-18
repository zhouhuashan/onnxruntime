#pragma once
#include "core/common/common.h"
#include "core/framework/allocation_planner.h"

namespace Lotus {
struct MemoryBlock {
  size_t offset_{0};
  size_t size_{0};

  MemoryBlock() = default;
  MemoryBlock(size_t offset, size_t size) : offset_(offset), size_(size) {}
};

class MemoryPattern {
  friend class MemPatternPlanner;

 public:
  MemoryPattern() = default;

  MemoryPattern(MemoryPattern&& rhs)
      : patterns_{std::move(rhs.patterns_)},
        peak_size_{std::move(rhs.peak_size_)} {}

  MemoryPattern& operator=(MemoryPattern&& rhs) {
    patterns_ = std::move(rhs.patterns_);
    peak_size_ = std::move(rhs.peak_size_);
    return *this;
  }

  size_t PeakSize() const {
    return peak_size_;
  }

  const MemoryBlock* GetBlock(int ml_value_idx) const {
    auto it = patterns_.find(ml_value_idx);
    if (it == patterns_.end())
      return nullptr;
    else
      return &it->second;
  }

 private:
  // allow move
  LOTUS_DISALLOW_COPY_AND_ASSIGN(MemoryPattern);

  std::unordered_map<int, MemoryBlock> patterns_;
  size_t peak_size_{0};
};

struct MemoryPatternGroup {
  std::vector<AllocatorInfo> locations;
  std::vector<MemoryPattern> patterns;

  const MemoryPattern* GetPatterns(const AllocatorInfo& location) const {
    for (int i = 0; i < locations.size(); i++)
      if (locations[i] == location) {
        return &patterns[i];
      }
    return nullptr;
  }
};
}  // namespace Lotus

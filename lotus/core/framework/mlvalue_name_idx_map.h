#pragma once

#include <atomic>
#include <unordered_map>

#include "core/common/common.h"

namespace onnxruntime {
class MLValueNameIdxMap {
 public:
  using const_iterator = typename std::unordered_map<std::string, int>::const_iterator;

  MLValueNameIdxMap() = default;

  // Add MLValue name to map and return index associated with it.
  // If entry already existed the existing index value is returned.
  int Add(const std::string& name) {
    int idx;
    common::Status status = GetIdx(name, idx);

    if (!status.IsOK()) {
      idx = mlvalue_max_idx_++;
      map_.insert({name, idx});
    }

    return idx;
  }

  common::Status GetIdx(const std::string& name, int& idx) const {
    idx = -1;

    auto it = map_.find(name);
    if (it == map_.end()) {
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "Could not find MLValue with name: ", name);
    }

    idx = it->second;
    return common::Status::OK();
  }

  size_t Size() const { return map_.size(); };
  int MaxIdx() const { return mlvalue_max_idx_; }

  const_iterator begin() const noexcept { return map_.cbegin(); }
  const_iterator end() const noexcept { return map_.cend(); }

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(MLValueNameIdxMap);

  // using std::atomic so each call to Add is guaranteed to get a unique index number.
  // we could use a raw int at the cost of thread safety, but as we populate the map during
  // initialization the performance cost of std::atomic is acceptable.
  std::atomic<int> mlvalue_max_idx_{0};
  std::unordered_map<std::string, int> map_;
};

}  // namespace onnxruntime

#pragma once
#include "core/common/common.h"
#include "core/framework/mem_pattern_planner.h"
#include "core/framework/allocation_planner.h"
#include <vector>
#include <memory>
#include <mutex>

namespace Lotus {
struct SequentialExecutionPlan;

class MLValuePatternPlanner {
 public:
  explicit MLValuePatternPlanner(const SequentialExecutionPlan& execution_plan);

  Common::Status TraceAllocation(int ml_value_idx, size_t size) {
    auto location = execution_planner_.allocation_plan[ml_value_idx].location;
    auto it = planner_map_.find(location);
    if (it == planner_map_.end()) {
      return Common::Status(Common::LOTUS, Common::INVALID_ARGUMENT);
    }

    std::lock_guard<std::mutex> lock(lock_);
    it->second->TraceAllocation(ml_value_idx, size);
    return Common::Status::OK();
  }

  Common::Status TraceFree(int ml_value_index) {
    auto location = execution_planner_.allocation_plan[ml_value_index].location;
    auto it = planner_map_.find(location);
    if (it == planner_map_.end()) {
      return Common::Status(Common::LOTUS, Common::INVALID_ARGUMENT);
    }

    std::lock_guard<std::mutex> lock(lock_);
    it->second->TraceFree(ml_value_index);
    return Common::Status::OK();
  }

  Common::Status GeneratePatterns(MemoryPatternGroup* out) {
    if (!out)
      return Common::Status(Common::LOTUS, Common::INVALID_ARGUMENT);

    std::lock_guard<std::mutex> lock(lock_);
    for (auto& it : planner_map_) {
      out->locations.push_back(it.first);
      out->patterns.push_back(it.second->GenerateMemPattern());
    }

    return Common::Status::OK();
  }

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(MLValuePatternPlanner);

  mutable std::mutex lock_;
  std::map<AllocatorInfo, MemPatternPlanner*> planner_map_;
  std::vector<std::unique_ptr<MemPatternPlanner> > pattern_planners_;
  const SequentialExecutionPlan& execution_planner_;
};
}  // namespace Lotus

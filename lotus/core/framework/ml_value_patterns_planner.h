#pragma once
#include "core/framework/mem_pattern_planner.h"
#include <mutex>

namespace Lotus {
class SessionSessionState;

class MLValuePatternPlanner {
 public:
  MLValuePatternPlanner(const SessionState& session_state);

  Status TraceAllocation(int ml_value_idx, size_t size) {
    auto location = execution_planner_->allocation_plan[ml_value_idx].location;
    auto it = planner_map_.find(location);
    if (it == planner_map_.end()) {
      return Status(LOTUS, INVALID_ARGUMENT);
    }

    std::lock_guard<std::mutex> lock(lock_);
    it->second->TraceAllocation(ml_value_idx, size);
    return Status::OK();
  }

  Status TraceFree(int ml_value_index) {
    auto location = execution_planner_->allocation_plan[ml_value_index].location;
    auto it = planner_map_.find(location);
    if (it == planner_map_.end()) {
      return Status(LOTUS, INVALID_ARGUMENT);
    }

    std::lock_guard<std::mutex> lock(lock_);
    it->second->TraceFree(ml_value_index);
    return Status::OK();
  }

  Status GeneratePatterns(MemoryPatternGroup* out) {
    if (!out)
      return Status(LOTUS, INVALID_ARGUMENT);

    std::lock_guard<std::mutex> lock(lock_);
    for (auto it = planner_map_.begin(); it != planner_map_.end(); it++) {
      out->locations.push_back(it->first);
      MemoryPattern p;
      LOTUS_RETURN_IF_ERROR(it->second->GenerateMemPattern(&p));
      out->patterns.push_back(p);
    }
    return Status::OK();
  }

 private:
  mutable std::mutex lock_;
  std::map<AllocatorInfo, MemPatternPlanner*> planner_map_;
  std::vector<std::unique_ptr<MemPatternPlanner> > pattern_planners_;
  const SequentialExecutionPlan* execution_planner_;
};
}  // namespace Lotus

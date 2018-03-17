#include "allocation_planner.h"

namespace Lotus {

std::pair<Status, unique_ptr<SequentialExecutionPlan>> SequentialPlanner::CreatePlan(const SessionState& session_state) {
  (session_state);
  // TODO
  return std::make_pair(Status(), unique_ptr<SequentialExecutionPlan>());
}

}  // namespace Lotus

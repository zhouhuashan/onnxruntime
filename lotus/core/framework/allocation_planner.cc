#include "allocation_planner.h"
#include "core/framework/session_state.h"

namespace Lotus {

Status SequentialPlanner::CreatePlan(const SessionState& session_state,
                                     SequentialExecutionPlan* plan) {
  UNUSED_PARAMETER(session_state);
  UNUSED_PARAMETER(plan);
  // TODO
  return Status::OK();
}
}  // namespace Lotus

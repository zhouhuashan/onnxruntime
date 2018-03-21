#include "allocation_planner.h"

namespace Lotus {

Status SequentialPlanner::CreatePlan(const SessionState& session_state,
                                     SequentialExecutionPlan* plan) {
  UNUSED_PARAMETER(session_state);
  UNUSED_PARAMETER(plan);
  // TODO
  return Status::OK();
}

Status DummyPlanner::CreatePlan(const SessionState& session_state,
                                SequentialExecutionPlan* plan) {
  size_t num_mlvalues = session_state.GetMaxMLValueIdx() + 1;
  // init allocation plan
  // all the values will be set as kAllocate except weights.
  // Weights are set as kAllocateStatically;
  plan->allocation_plan.resize(num_mlvalues);
  for (int i = 0; i < num_mlvalues; i++) {
    plan->allocation_plan[i].alloc_kind = AllocKind::kAllocate;
  }

  auto graph = session_state.GetGraph();
  auto& weights = graph->GetAllInitializedTensors();
  int index = 0;
  for (auto it = weights.begin(); it != weights.end(); it++) {
    LOTUS_RETURN_IF_ERROR(session_state.GetMLValueIdx(it->first, &index));
    plan->allocation_plan[index].alloc_kind = AllocKind::kAllocateStatically;
  }

  //setup execution plan
  std::vector<LotusIR::NODEINDEX>* order;
  LOTUS_RETURN_IF_ERROR(const_cast<Graph*>(graph)->GetNodesInTopologicalOrder(&order));
  for (int i = 0; i < order->size(); i++) {
    if (graph->IsSinkNode((*order)[i]) || graph->IsSourceNode((*order)[i]))
      continue;
    SequentialExecutionPlan::NodeExecutionPlan p;
    p.node_index = (*order)[i];
    // the dummy plan won't free any tensor after execute a node.
    // so set the start and end to 0 means nothing to free
    p.free_from_index = 1;
    p.free_to_index = 0;
    plan->execution_plan.push_back(p);
  }
  // because there is nothing to free in the middle of execution
  // to_be_freed vector will be left as empty.
  return Status::OK();
}

}  // namespace Lotus

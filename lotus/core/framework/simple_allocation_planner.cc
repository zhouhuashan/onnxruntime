#include "core/framework/session_state.h"
#include "core/framework/allocation_planner.h"
#include "core/graph/utils.h"

namespace Lotus {

Status FillType(const LotusIR::NodeArg& arg, const SessionState& session_state, SequentialExecutionPlan* plan) {
  int index;
  LOTUS_RETURN_IF_ERROR(session_state.GetMLValueIdx(arg.Name(), &index));
  auto type = DataTypeImpl::TypeFromProto(
      onnx::Utils::DataTypeUtils::ToTypeProto(arg.Type()));
  if (plan->allocation_plan[index].value_type != nullptr &&
      plan->allocation_plan[index].value_type != type)
    return Status(LOTUS, FAIL, "Found MLValue has type conflict.");

  plan->allocation_plan[index].value_type = type;

  return Status::OK();
}

Status SimpleAllocationPlanner::CreatePlan(const SessionState& session_state,
                                           SequentialExecutionPlan* plan) {
  size_t num_mlvalues = session_state.GetMaxMLValueIdx() + 1;
  // init allocation plan
  // all the values will be set as kAllocate except weights.
  // Weights are set as kAllocateStatically;
  plan->allocation_plan.resize(num_mlvalues);
  for (int i = 0; i < num_mlvalues; i++) {
    plan->allocation_plan[i].alloc_kind = AllocKind::kAllocate;
    // TODO: resolve the correct location of the values.
    auto cpu_provider = session_state.GetExecutionProvider(LotusIR::kCpuExecutionProvider);
    LOTUS_ENFORCE(cpu_provider);
    plan->allocation_plan[i].location = cpu_provider->GetAllocator()->Info();
  }

  auto graph = session_state.GetGraph();
  // iterate all the values in the graph to assign the correct type.
  for (auto& node : graph->Nodes()) {
    if (graph->IsSinkNode(node) || graph->IsSourceNode(node))
      continue;

    for (gsl::not_null<const LotusIR::NodeArg*> input_def : node.InputDefs()) {
      FillType(*input_def, session_state, plan);
    }

    for (gsl::not_null<const LotusIR::NodeArg*> output_def : node.OutputDefs()) {
      FillType(*output_def, session_state, plan);
    }
  }

  auto& weights = graph->GetAllInitializedTensors();
  int index = 0;
  for (auto it = weights.begin(); it != weights.end(); it++) {
    LOTUS_RETURN_IF_ERROR(session_state.GetMLValueIdx(it->first, &index));
    plan->allocation_plan[index].alloc_kind = AllocKind::kAllocateStatically;
  }

  //setup execution plan
  const std::vector<LotusIR::NodeIndex>* order;
  LOTUS_RETURN_IF_ERROR(graph->GetNodesInTopologicalOrder(&order));
  for (int i = 0; i < order->size(); i++) {
    if (graph->IsSinkNode((*order)[i]) || graph->IsSourceNode((*order)[i]))
      continue;
    // the dummy plan won't free any tensor after execute a node.
    // so set the start and end to 0 means nothing to free
    SequentialExecutionPlan::NodeExecutionPlan p((*order)[i]);
    plan->execution_plan.push_back(p);
  }

  // because there is nothing to free in the middle of execution
  // to_be_freed vector will be left as empty.
  return Status::OK();
}

}  // namespace Lotus

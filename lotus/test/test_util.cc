#include "test/test_utils.h"

namespace Lotus {
namespace Test {

void SetupState(SessionState& state,
                const std::vector<LotusIR::NodeArg*>& input_defs,
                const std::vector<LotusIR::NodeArg*>& output_defs) {
  int idx = 0;
  for (auto& elem : input_defs) {
    state.AddMLValueNameIdx(elem->Name(), idx++);
  }
  for (auto& elem : output_defs) {
    state.AddMLValueNameIdx(elem->Name(), idx++);
  }

  std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan = std::make_unique<SequentialExecutionPlan>();
  // TODO change SimpleAllocationPlanner to use SequentialPlanner; Simple exists for testing only.
  SimpleAllocationPlanner::CreatePlan(state, p_seq_exec_plan.get());
  state.SetExecutionPlan(std::move(p_seq_exec_plan));
}

void FillFeedsAndOutputNames(const std::vector<LotusIR::NodeArg*>& input_defs,
                             const std::vector<LotusIR::NodeArg*>& output_defs,
                             std::unordered_map<std::string, MLValue>& feeds,
                             std::vector<std::string>& output_names) {
  for (auto& elem : input_defs) {
    feeds.insert(std::make_pair(elem->Name(), MLValue()));
  }
  for (auto& elem : output_defs) {
    output_names.push_back(elem->Name());
  }
}

}  // namespace Test
}  // namespace Lotus

#include "test/providers/provider_test_utils.h"

#include <memory>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "test/test_utils.h"

using namespace Lotus::Logging;

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

  // if you want to use a non-default logger this has to happen at a scope where ownership of that
  // logger makes sense. below is example code to do that, which should run after a call to SetupState
  //
  // std::unique_ptr<Logging::Logger> logger{DefaultLoggingManager().CreateLogger("MyLogId")};
  // state.SetLogger(*logger);

  state.SetLogger(DefaultLoggingManager().DefaultLogger());
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

template <>
void OpTester::Check<float>(const Data& output_data, Tensor& output_tensor, size_t size) {
  auto* expected = reinterpret_cast<const float*>(output_data.data_.get());
  auto* output = output_tensor.Data<float>();
  for (int i = 0; i < size; ++i) {
    EXPECT_NEAR(expected[i], output[i], 0.001f);
  }
}

template <>
void OpTester::Check<bool>(const Data& output_data, Tensor& output_tensor, size_t size) {
  auto* expected = reinterpret_cast<const bool*>(output_data.data_.get());
  auto* output = output_tensor.Data<bool>();
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
}

}  // namespace Test
}  // namespace Lotus

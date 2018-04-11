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

// These have to be defined before Run() since Run() tries to instantiate them
template <>
void OpTester::Check<float>(const Data& output_data, Tensor& output_tensor, size_t size) {
  auto* expected = reinterpret_cast<const float*>(output_data.data_.get());
  auto* output = output_tensor.Data<float>();
  for (int i = 0; i < size; ++i) {
    if (std::isinf(expected[i]))  // Test infinity for equality
      EXPECT_EQ(expected[i], output[i]);
    else
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

OpTester::~OpTester() {
#if _DEBUG
  if (!run_called_) {
    std::cerr << "Someone forgot to call OpTester::Run()" << std::endl;
    __debugbreak();
  }
#endif
}

void OpTester::Run() {
#if _DEBUG
  run_called_ = true;
#endif

  // Generate the input & output def lists
  std::vector<LotusIR::NodeArg*> input_defs;
  std::vector<LotusIR::NodeArg*> output_defs;

  for (auto& data : input_data_)
    input_defs.push_back(&data.def_);

  for (auto& data : output_data_)
    output_defs.push_back(&data.def_);

  // Create a simple model
  LotusIR::Model model{"test"};
  LotusIR::Graph* graph = model.MainGraph();
  graph->AddNode("node1", op_, op_, input_defs, output_defs);
  graph->Resolve();

  SessionState state;
  state.SetGraph(graph);
  SetupState(state, input_defs, output_defs);

  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(input_defs, output_defs, feeds, output_names);

  std::shared_ptr<ExecutionFrame> frame{TestUtils::CreateSingleNodeCPUExecutionFrame(state, feeds, output_names)};

  auto& node = *graph->GetNode(graph->NumberOfNodes() - 1);

  // Add the attributes if any
  for (auto& add_attribute_fn : add_attribute_funcs_)
    add_attribute_fn(node);

  node.SetExecutionProvider(LotusIR::kCpuExecutionProvider);

  // Setup the op in the node
  AllocatorInfo allocator_info{CPU, Lotus::AllocatorType::kArenaAllocator};
  KernelDef kernel_def;
  unique_ptr<OpKernel> kernel;
  Status status = KernelRegistry::Instance().CreateKernel(node, allocator_info, nullptr, &kernel);
  LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());

  // Hookup the inputs and outputs
  unsigned index = 0;
  for (auto& input : input_data_) {
    status = frame->AllocateTensorWithSelfOwnBuffer(
        index, input.data_type_, AllocatorManager::Instance().GetArena(CPU).Info(), input.shape_);
    LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());
    // For inputs we have data to initialize with, so copy it into the buffer
    auto* tensor = frame->GetMutableValue<Tensor>(index);
    void* buffer = tensor->MutableDataRaw(input.data_type_);
    memcpy(buffer, input.data_.get(), input.data_size_in_bytes_);
    index++;
  }

  // Note, index isn't reset here since outputs are indexed after inputs
  for (auto& output : output_data_) {
    status = frame->AllocateTensorWithSelfOwnBuffer(
        index++, output.data_type_, AllocatorManager::Instance().GetArena(CPU).Info(), output.shape_);
    LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());
  }

  // Run the model
  OpKernelContext kernel_ctx(frame.get(), kernel.get(), DefaultLoggingManager().DefaultLogger());
  status = kernel->Compute(&kernel_ctx);
  LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());

  // Verify the outputs
  index = 0;
  for (auto& output : output_data_) {
    auto& outputTensor = *kernel_ctx.Output(index++, output.shape_);
    auto size = output.shape_.Size();
    LOTUS_ENFORCE(output.shape_ == outputTensor.Shape(), "Output shape did not match expected output shape");

    // Dispatch on the type
    if (output.data_type_ == DataTypeImpl::GetType<float>())
      Check<float>(output, outputTensor, size);
    else if (output.data_type_ == DataTypeImpl::GetType<bool>())
      Check<bool>(output, outputTensor, size);
  }
}

}  // namespace Test
}  // namespace Lotus

#ifndef TEST_TEST_UTILS_H
#define TEST_TEST_UTILS_H

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"

namespace Lotus {
namespace Test {

void SetupState(SessionState& state,
                const std::vector<LotusIR::NodeArg*>& input_defs,
                const std::vector<LotusIR::NodeArg*>& output_defs);

void FillFeedsAndOutputNames(const std::vector<LotusIR::NodeArg*>& input_defs,
                             const std::vector<LotusIR::NodeArg*>& output_defs,
                             std::unordered_map<std::string, MLValue>& feeds,
                             std::vector<std::string>& output_names);

class TestUtils {
  typedef std::shared_ptr<ExecutionFrame> ExecutionFramePtr;

 public:
  static ExecutionFramePtr CreateSingleNodeCPUExecutionFrame(
      const SessionState& session_state,
      std::unordered_map<std::string, MLValue> feeds,
      const std::vector<std::string> output_names) {
    return std::make_shared<ExecutionFrame>(
        feeds,
        output_names,
        session_state);
  }

  template <typename T>
  static Status PrepareTensor(const int index,
                              ExecutionFramePtr frame,
                              const std::vector<int64_t>& dims,
                              const std::vector<T>* value) {
    auto status = frame->AllocateTensorWithSelfOwnBuffer(
        index,
        DataTypeImpl::GetType<T>(),
        AllocatorManager::Instance()->GetArena(CPU).Info(),
        TensorShape(dims));
    if (!status.IsOK())
      return status;
    if (value) {
      auto tensor = frame->get_mutable_value<Tensor>(index);
      LOTUS_ENFORCE(size_t(tensor->shape().Size()) == value->size(), "Number of input values doesn't match tensor size");
      T* buffer = tensor->mutable_data<T>();
      for (int i = 0; i < value->size(); i++)
        buffer[i] = (*value)[i];
    }
    return Status::OK();
  }

  template <typename T>
  static Status PrepareIthInput(const LotusIR::Node& node,
                                const int i,
                                ExecutionFramePtr frame,
                                const std::vector<int64_t>& dims,
                                const std::vector<T>* value = nullptr) {
    LOTUS_ENFORCE(i >= 0 && i < node.InputDefs().size());
    return PrepareTensor(i, frame, dims, value);
  }

  template <typename T>
  static Status PrepareIthOutput(const LotusIR::Node& node,
                                 const int i,
                                 ExecutionFramePtr frame,
                                 const std::vector<int64_t>& dims,
                                 const std::vector<T>* value = nullptr) {
    LOTUS_ENFORCE(i >= 0 && i < node.OutputDefs().size());
    return PrepareTensor(i + (int)node.InputDefs().size(), frame, dims, value);
  }
};

struct TypeProto_Set : TypeProto {
  TypeProto_Set(TensorProto_DataType type) {
    mutable_tensor_type()->set_elem_type(type);
  }
};

extern TypeProto_Set s_typeProto_float;

struct TestModel {
  TestModel(const char* szName, const std::vector<LotusIR::NodeArg*>& inputDefs, const std::vector<LotusIR::NodeArg*>& outputDefs) {
    Graph()->AddNode("node1", szName, szName, inputDefs, outputDefs);

    Graph()->Resolve();
    state_.SetGraph(Graph());
    SetupState(state_, inputDefs, outputDefs);

    std::unordered_map<std::string, MLValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(inputDefs, outputDefs, feeds, output_names);
    frame_ = TestUtils::CreateSingleNodeCPUExecutionFrame(state_, feeds, output_names);
  }

  LotusIR::Graph* Graph() { return model_.MainGraph(); }
  LotusIR::Node& Node() { return *Graph()->GetNode(Graph()->NumberOfNodes() - 1); };
  auto& State() { return state_; }
  auto& Frame() { return frame_; }

 private:
  LotusIR::Model model_{"test", true};
  SessionState state_;
  std::shared_ptr<ExecutionFrame> frame_;
};

// To use SimpleFloatTest:
//  1. Create a TestModel
//  2. Add any attributes
//  3. Create a SimpleFloatTest
//  4. Add any inputs/outputs
//  5. Call SimpleFloatTest::Run with the expected output
template <template <typename> typename Op>
struct SimpleFloatTest {
  SimpleFloatTest(TestModel& model)
      : model_(model) {
  }

  template <size_t count>
  void Run(const std::vector<int64_t>& expectedDims, const float (&expected_vals)[count]) {
    OpKernelContext kernel_ctx(model_.Frame().get(), &kernel_);
    kernel_.compute(&kernel_ctx);
    auto& output = *kernel_ctx.output(0, TensorShape(expectedDims));
    Check(output, expected_vals);
  }

  template <size_t count>
  static void Check(Tensor& output, const float (&expected_vals)[count]) {
    LOTUS_ENFORCE(output.shape().Size() == count);
    const float* res = output.data<float>();
    for (int i = 0; i < count; ++i) {
      EXPECT_NEAR(expected_vals[i], res[i], 0.001f);
    }
  }

  void AddInput(const std::vector<int64_t>& dims, const std::vector<float>& values) {
    auto status = TestUtils::PrepareIthInput<float>(model_.Node(), inputCount_++, model_.Frame(), dims, &values);
    EXPECT_TRUE(status.IsOK());
  }

  void AddOutput(const std::vector<int64_t>& dims) {
    auto status = TestUtils::PrepareIthOutput<float>(model_.Node(), 0, model_.Frame(), dims, nullptr);
    EXPECT_TRUE(status.IsOK());
  }

  TestModel& model_;
  AllocatorInfo allocator_info_{"CPUAllocator", Lotus::AllocatorType::ArenaAllocator};
  KernelDef kernel_def_;
  OpKernelInfo info_{model_.Node(), allocator_info_, kernel_def_};
  Op<float> kernel_{info_};

  unsigned inputCount_{};
};

#define CREATE_NODE(op_name, inputs, outputs)                 \
  LotusIR::Model model("test");                               \
  LotusIR::Graph* graph = model.MainGraph();                  \
  graph->AddNode("node1", op_name, op_name, inputs, outputs); \
  LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);
}  // namespace Test
}  // namespace Lotus

#endif  // !CORE_TEST_TEST_UTIL_H

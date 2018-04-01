#pragma once

#include "core/common/logging/logging.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"

#include "test/test_utils.h"

#include "gtest/gtest.h"

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

// TODO(RyanHill): Remove once usage is switched over to s_typeProto<> type
struct TypeProto_Set : TypeProto {
  TypeProto_Set(TensorProto_DataType type) {
    mutable_tensor_type()->set_elem_type(type);
  }
};

// Function templates to translate C++ types into TensorProto_DataTypes
template <typename T>
constexpr TensorProto_DataType TypeToDataType();

template <>
constexpr TensorProto_DataType TypeToDataType<float>() { return TensorProto_DataType_FLOAT; }

template <>
constexpr TensorProto_DataType TypeToDataType<bool>() { return TensorProto_DataType_BOOL; }

template <typename T>
struct TTypeProto : TypeProto {
  TTypeProto() {
    mutable_tensor_type()->set_elem_type(TypeToDataType<T>());
  }
};

// Variable template for TensorProto_DataTypes, s_typeProto<float>
template <typename T>
const TTypeProto<T> s_typeProto;

// To use OpTester:
//  1. Create one with a name
//  2. Call AddAttribute with any attributes
//  3. Call AddInput for all the inputs
//  4. Call AddOutput with all expected outputs
//  5. Call Run with the fully defined Op as the template parameter
// Currently only works for float & bool tensors
// See current usage for an example, should be self explanatory
struct OpTester {
  OpTester(const char* szName) : szName_(szName) {}

  template <typename T>
  void AddInput(const char* szName, const std::vector<int64_t>& dims, const std::initializer_list<T>& values) {
    AddData(inputData_, szName, dims, values);
  }

  template <typename T>
  void AddOutput(const char* szName, const std::vector<int64_t>& dims, const std::initializer_list<T>& expectedValues) {
    AddData(outputData_, szName, dims, expectedValues);
  }

  template <typename T>
  void AddAttribute(const char* szName, T value) {
    // Copy the attribute data for now, since we have to add them at a later point
    auto pData = std::make_unique<uint8_t[]>(sizeof(T));
    memcpy(pData.get(), &value, sizeof(T));
    // Use a lambda to generate a type safe AddAttribute call later
    attributes_.push_back({szName, std::move(pData), [](Node& node, Attribute& attribute) { EXPECT_TRUE(node.AddAttribute(attribute.szName_, *reinterpret_cast<T*>(attribute.data_.get()))); }});
  }

  template <typename Op>
  void Run() {
    // Generate the input & output def lists
    std::vector<LotusIR::NodeArg*> pinputDefs, poutputDefs;
    for (auto& data : inputData_)
      pinputDefs.push_back(&data.def_);
    for (auto& data : outputData_)
      poutputDefs.push_back(&data.def_);

    // Create a simple model
    LotusIR::Model model{"test"};
    LotusIR::Graph* graph = model.MainGraph();
    graph->AddNode("node1", szName_, szName_, pinputDefs, poutputDefs);
    graph->Resolve();

    SessionState state;
    state.SetGraph(graph);
    SetupState(state, pinputDefs, poutputDefs);

    std::unordered_map<std::string, MLValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(pinputDefs, poutputDefs, feeds, output_names);

    std::shared_ptr<ExecutionFrame> frame{TestUtils::CreateSingleNodeCPUExecutionFrame(state, feeds, output_names)};

    auto& node = *graph->GetNode(graph->NumberOfNodes() - 1);

    // Add the attributes if any
    for (auto& attribute : attributes_)
      attribute.AddAttribute_(node, attribute);

    // Setup the op in the node
    AllocatorInfo allocator_info{CPU, Lotus::AllocatorType::ArenaAllocator};
    KernelDef kernel_def;
    OpKernelInfo info{node, allocator_info, kernel_def};
    Op kernel{info};

    // Hookup the inputs and outputs
    unsigned index = 0;
    for (auto& input : inputData_) {
      auto status = frame->AllocateTensorWithSelfOwnBuffer(index, input.dataType_, AllocatorManager::Instance()->GetArena(CPU).Info(), input.shape_);
      // For inputs we have data to initialize with, so copy it into the buffer
      auto* tensor = frame->get_mutable_value<Tensor>(index);
      void* buffer = tensor->mutable_data_raw(input.dataType_);
      memcpy(buffer, input.data_.get(), input.dataSize_);
      index++;
    }

    index = 0;
    for (auto& output : outputData_) {
      auto status = frame->AllocateTensorWithSelfOwnBuffer(index++, output.dataType_, AllocatorManager::Instance()->GetArena(CPU).Info(), output.shape_);
    }

    // Run the model
    OpKernelContext kernel_ctx(frame.get(), &kernel, DefaultLoggingManager().DefaultLogger());
    Common::Status status = kernel.compute(&kernel_ctx);
    LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());

    // Verify the outputs
    index = 0;
    for (auto& output : outputData_) {
      auto& outputTensor = *kernel_ctx.output(index++, output.shape_);
      auto size = output.shape_.Size();

      // Dispatch on the type
      if (output.dataType_ == DataTypeImpl::GetType<float>())
        Check<float>(output, outputTensor, size);
      else if (output.dataType_ == DataTypeImpl::GetType<bool>())
        Check<bool>(output, outputTensor, size);
    }
  }

 private:
  struct Data {
    LotusIR::NodeArg def_;
    TensorShape shape_;
    std::unique_ptr<uint8_t[]> data_;
    size_t dataSize_;
    MLDataType dataType_;
  };

  template <typename T>
  void AddData(std::vector<Data>& data, const char* szName, const std::vector<int64_t>& dims, const std::initializer_list<T>& values) {
    LOTUS_ENFORCE(TensorShape(dims).Size() == values.size(), "Number of input values doesn't match tensor size");
    auto size = values.size() * sizeof(T);
    auto pData = std::make_unique<uint8_t[]>(size);
    memcpy(pData.get(), values.begin(), size);
    data.push_back({{szName, &s_typeProto<T>}, dims, std::move(pData), size, DataTypeImpl::GetType<T>()});
  }

  struct Attribute {
    const char* szName_;
    std::unique_ptr<uint8_t[]> data_;
    void (*AddAttribute_)(Node& node, Attribute& attribute);
  };

  // Templatize the check function on type so we can compare properly (specializations defined in provider_test_utils.cc)
  template <typename T>
  void Check(const Data& outputData, Tensor& outputTensor, size_t size);

  const char* szName_;
  std::vector<Data> inputData_, outputData_;
  std::vector<Attribute> attributes_;
};

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
  LotusIR::Model model_{"test"};
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
    OpKernelContext kernel_ctx(model_.Frame().get(), &kernel_, DefaultLoggingManager().DefaultLogger());
    Common::Status status = kernel_.compute(&kernel_ctx);
    LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());
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
  AllocatorInfo allocator_info_{CPU, Lotus::AllocatorType::ArenaAllocator};
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

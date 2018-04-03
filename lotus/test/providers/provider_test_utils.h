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
        AllocatorManager::Instance().GetArena(CPU).Info(),
        TensorShape(dims));
    if (!status.IsOK())
      return status;
    if (value) {
      auto tensor = frame->GetMutableValue<Tensor>(index);
      LOTUS_ENFORCE(size_t(tensor->Shape().Size()) == value->size(), "Number of input values doesn't match tensor size");
      T* buffer = tensor->MutableData<T>();
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

  // We have an initializer_list and vector version of the Add functions because std::vector is specialized for
  // bools and we can't get the raw data out. So those cases must use an initializer_list
  template <typename T>
  void AddInput(const char* szName, const std::vector<int64_t>& dims, const std::initializer_list<T>& values) {
    AddData(inputData_, szName, dims, values.begin(), values.size());
  }

  template <typename T>
  void AddInput(const char* szName, const std::vector<int64_t>& dims, const std::vector<T>& values) {
    AddData(inputData_, szName, dims, values.data(), values.size());
  }

  template <typename T>
  void AddOutput(const char* szName, const std::vector<int64_t>& dims, const std::initializer_list<T>& expectedValues) {
    AddData(outputData_, szName, dims, expectedValues.begin(), expectedValues.size());
  }

  template <typename T>
  void AddOutput(const char* szName, const std::vector<int64_t>& dims, const std::vector<T>& expectedValues) {
    AddData(outputData_, szName, dims, expectedValues.data(), expectedValues.size());
  }

  template <typename T>
  void AddAttribute(const char* szName, T value) {
    // Generate a the proper AddAttribute call for later
    addAttributeFns_.emplace_back([szName, value = std::move(value)](LotusIR::Node& node) { node.AddAttribute(szName, value); });
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
    for (auto& addAttributeFn : addAttributeFns_)
      addAttributeFn(node);

    // Setup the op in the node
    AllocatorInfo allocator_info{CPU, Lotus::AllocatorType::kArenaAllocator};
    KernelDef kernel_def;
    OpKernelInfo info{node, allocator_info, kernel_def};
    Op kernel{info};

    // Hookup the inputs and outputs
    unsigned index = 0;
    for (auto& input : inputData_) {
      auto status = frame->AllocateTensorWithSelfOwnBuffer(
          index, input.dataType_, AllocatorManager::Instance().GetArena(CPU).Info(), input.shape_);
      // For inputs we have data to initialize with, so copy it into the buffer
      auto* tensor = frame->GetMutableValue<Tensor>(index);
      void* buffer = tensor->MutableDataRaw(input.dataType_);
      memcpy(buffer, input.data_.get(), input.dataSizeInBytes_);
      index++;
    }

    // Note, index isn't reset here since outputs are indexed after inputs
    for (auto& output : outputData_) {
      auto status = frame->AllocateTensorWithSelfOwnBuffer(
          index++, output.dataType_, AllocatorManager::Instance().GetArena(CPU).Info(), output.shape_);
    }

    // Run the model
    OpKernelContext kernel_ctx(frame.get(), &kernel, DefaultLoggingManager().DefaultLogger());
    Common::Status status = kernel.Compute(&kernel_ctx);
    LOTUS_ENFORCE(status.IsOK(), status.ErrorMessage());

    // Verify the outputs
    index = 0;
    for (auto& output : outputData_) {
      auto& outputTensor = *kernel_ctx.Output(index++, output.shape_);
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
    size_t dataSizeInBytes_;
    MLDataType dataType_;
  };

  template <typename T>
  void AddData(std::vector<Data>& data, const char* szName, const std::vector<int64_t>& dims, const T* values, size_t valuesCount) {
    static_assert(std::is_trivial<T>::value, "Only works on trivial types (where byte copies of the values are safe)");
    LOTUS_ENFORCE(TensorShape(dims).Size() == valuesCount, "Number of input values doesn't match tensor size");
    auto sizeInBytes = valuesCount * sizeof(T);
    auto pData = std::make_unique<uint8_t[]>(sizeInBytes);
    memcpy(pData.get(), values, sizeInBytes);
    data.push_back({{szName, &s_typeProto<T>}, dims, std::move(pData), sizeInBytes, DataTypeImpl::GetType<T>()});
  }

  // Templatize the check function on type so we can compare properly (specializations defined in provider_test_utils.cc)
  template <typename T>
  void Check(const Data& output_data, Tensor& output_tensor, size_t size);

  const char* szName_;
  std::vector<Data> inputData_, outputData_;
  std::vector<std::function<void(LotusIR::Node& node)>> addAttributeFns_;
};

#define CREATE_NODE(op_name, inputs, outputs)                 \
  LotusIR::Model model("test");                               \
  LotusIR::Graph* graph = model.MainGraph();                  \
  graph->AddNode("node1", op_name, op_name, inputs, outputs); \
  LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);
}  // namespace Test
}  // namespace Lotus

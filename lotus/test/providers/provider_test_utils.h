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
    static std::vector<MLValue> outputs;
    return std::make_shared<ExecutionFrame>(
        feeds,
        output_names,
        outputs,
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

// Function templates to translate C++ types into TensorProto_DataTypes
template <typename T>
constexpr TensorProto_DataType TypeToDataType();

template <>
constexpr TensorProto_DataType TypeToDataType<float>() { return TensorProto_DataType_FLOAT; }

template <>
constexpr TensorProto_DataType TypeToDataType<int32_t>() { return TensorProto_DataType_INT32; }

template <>
constexpr TensorProto_DataType TypeToDataType<int64_t>() { return TensorProto_DataType_INT64; }

template <>
constexpr TensorProto_DataType TypeToDataType<bool>() { return TensorProto_DataType_BOOL; }

template <typename T>
struct TTypeProto : TypeProto {
  TTypeProto() {
    mutable_tensor_type()->set_elem_type(TypeToDataType<T>());
  }
};

// Variable template for TensorProto_DataTypes, s_typeProto<float>, etc..
template <typename T>
const TTypeProto<T> s_typeProto;

// To use OpTester:
//  1. Create one with the op name
//  2. Call AddAttribute with any attributes
//  3. Call AddInput for all the inputs
//  4. Call AddOutput with all expected outputs
//  5. Call Run
// Not all tensor types and output types are added, if a new input type is used, add it to the TypeToDataType list above
// for new output types, add a new specialization for Check<>
// See current usage for an example, should be self explanatory
struct OpTester {
  OpTester(const char* szOp) : szOp_(szOp) {}
  ~OpTester();

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
  void AddAttribute(std::string name, T value) {
    // Generate a the proper AddAttribute call for later
    addAttributeFns_.emplace_back([name = std::move(name), value = std::move(value)](LotusIR::Node& node) { node.AddAttribute(name, value); });
  }

  void Run();

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

  const char* szOp_;
  std::vector<Data> inputData_, outputData_;
  std::vector<std::function<void(LotusIR::Node& node)>> addAttributeFns_;
#if _DEBUG
  bool m_fRun{};
#endif
};

}  // namespace Test
}  // namespace Lotus

#pragma once

#include "core/common/logging/logging.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"

#include "test/test_environment.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <gsl/gsl_byte>

namespace Lotus {
namespace Test {

void FillFeedsAndOutputNames(const std::vector<LotusIR::NodeArg*>& input_defs,
                             const std::vector<LotusIR::NodeArg*>& output_defs,
                             std::unordered_map<std::string, MLValue>& feeds,
                             std::vector<std::string>& output_names);

class TestUtils {
  typedef std::shared_ptr<ExecutionFrame> ExecutionFramePtr;

 public:
  static ExecutionFramePtr CreateSingleNodeCPUExecutionFrame(const SessionState& session_state,
                                                             std::unordered_map<std::string, MLValue> feeds,
                                                             const std::vector<std::string> output_names) {
    static std::vector<MLValue> outputs;

    return std::make_shared<ExecutionFrame>(feeds,
                                            output_names,
                                            outputs,
                                            session_state);
  }

  template <typename T>
  static Status PrepareTensor(const int index,
                              ExecutionFramePtr frame,
                              const std::vector<int64_t>& dims,
                              const std::vector<T>* value) {
    auto status = frame->AllocateTensorWithSelfOwnBuffer(index,
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
constexpr TensorProto_DataType TypeToDataType<double>() { return TensorProto_DataType_DOUBLE; }

template <>
constexpr TensorProto_DataType TypeToDataType<int32_t>() { return TensorProto_DataType_INT32; }

template <>
constexpr TensorProto_DataType TypeToDataType<int64_t>() { return TensorProto_DataType_INT64; }

template <>
constexpr TensorProto_DataType TypeToDataType<bool>() { return TensorProto_DataType_BOOL; }

template <>
constexpr TensorProto_DataType TypeToDataType<int16_t>() { return TensorProto_DataType_INT16; }

template <>
constexpr TensorProto_DataType TypeToDataType<uint8_t>() { return TensorProto_DataType_UINT8; }

template <>
constexpr TensorProto_DataType TypeToDataType<uint16_t>() { return TensorProto_DataType_UINT16; }

template <>
constexpr TensorProto_DataType TypeToDataType<uint32_t>() { return TensorProto_DataType_UINT32; }

template <>
constexpr TensorProto_DataType TypeToDataType<uint64_t>() { return TensorProto_DataType_UINT64; }

template <>
constexpr TensorProto_DataType TypeToDataType<std::string>() { return TensorProto_DataType_STRING; }

template <typename T>
struct TTypeProto : TypeProto {
  TTypeProto() {
    mutable_tensor_type()->set_elem_type(TypeToDataType<T>());
  }
};

// Variable template for TensorProto_DataTypes, s_type_proto<float>, etc..
template <typename T>
const TTypeProto<T> s_type_proto;

//TypeProto for map<TKey, TVal>
template <typename TKey, typename TVal>
struct MTypeProto : TypeProto {
  MTypeProto() {
    mutable_map_type()->set_key_type(TypeToDataType<TKey>());
    mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TypeToDataType<TVal>());
    mutable_map_type()->mutable_value_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  }
};

template <typename TKey, typename TVal>
const MTypeProto<TKey, TVal> s_map_type_proto;

//TypeProto for vector<map<TKey, TVal>>
template <typename TKey, typename TVal>
struct VectorOfMapTypeProto : TypeProto {
  VectorOfMapTypeProto() {
    auto* map_type = mutable_sequence_type()->mutable_elem_type()->mutable_map_type();
    map_type->set_key_type(TypeToDataType<TKey>());
    map_type->mutable_value_type()->mutable_tensor_type()->set_elem_type(TypeToDataType<TVal>());
    map_type->mutable_value_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  }
};

template <typename TKey, typename TVal>
const VectorOfMapTypeProto<TKey, TVal> s_vec_map_type_proto;

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
  OpTester(const char* op, const char* domain = LotusIR::kOnnxDomain) : op_(op), domain_(domain) {}
  ~OpTester();

  // We have an initializer_list and vector version of the Add functions because std::vector is specialized for
  // bool and we can't get the raw data out. So those cases must use an initializer_list
  template <typename T>
  void AddInput(const char* name, const std::vector<int64_t>& dims, const std::initializer_list<T>& values) {
    AddData(input_data_, name, dims, values.begin(), values.size());
  }

  template <typename T>
  void AddInput(const char* name, const std::vector<int64_t>& dims, const std::vector<T>& values) {
    AddData(input_data_, name, dims, values.data(), values.size());
  }

  template <typename TKey, typename TVal>
  void AddInput(const char* name, const std::map<TKey, TVal>& val) {
    std::unique_ptr<std::map<TKey, TVal>> ptr = make_unique<std::map<TKey, TVal>>(val);
    MLValue value;
    value.Init(ptr.release(),
               DataTypeImpl::GetType<std::map<TKey, TVal>>(),
               DataTypeImpl::GetType<std::map<TKey, TVal>>()->GetDeleteFunc());
    input_data_.push_back({{name, &s_map_type_proto<TKey, TVal>}, value});
  }

  template <typename T>
  void AddOutput(const char* name, const std::vector<int64_t>& dims, const std::initializer_list<T>& expected_values) {
    AddData(output_data_, name, dims, expected_values.begin(), expected_values.size());
  }

  template <typename T>
  void AddOutput(const char* name, const std::vector<int64_t>& dims, const std::vector<T>& expected_values) {
    AddData(output_data_, name, dims, expected_values.data(), expected_values.size());
  }

  // Add non tensor output
  template <typename TKey, typename TVal>
  void AddOutput(const char* name, const std::vector<std::map<TKey, TVal>>& val) {
    auto ptr = make_unique<std::vector<std::map<TKey, TVal>>>(val);
    MLValue ml_value;
    ml_value.Init(ptr.release(),
                  DataTypeImpl::GetType<std::vector<std::map<TKey, TVal>>>(),
                  DataTypeImpl::GetType<std::vector<std::map<TKey, TVal>>>()->GetDeleteFunc());
    output_data_.push_back({{name, &s_vec_map_type_proto<TKey, TVal>}, ml_value});
  }

  void SetOutputAbsErr(const char* name, float v);
  void SetOutputRelErr(const char* name, float v);

  template <typename T>
  void AddAttribute(std::string name, T value) {
    // Generate a the proper AddAttribute call for later
    add_attribute_funcs_.emplace_back(
        [name = std::move(name), value = std::move(value)](LotusIR::Node& node) { node.AddAttribute(name, value); });
  }

  void Run(bool expect_failure = false, const std::string& expected_failure_string = "");

 private:
  // unfortunately std::optional is in C++17 so use a miniversion of it
  template <typename T>
  class optional {
   public:
    optional(T v) : has_value_(true), value_(v) {}
    optional() : has_value_(false) {}
    bool has_value() const { return has_value_; }
    const T& value() const {
      LOTUS_ENFORCE(has_value_);
      return value_;
    }

   private:
    bool has_value_;
    T value_;
  };

  struct Data {
    LotusIR::NodeArg def_;
    MLValue data_;
    optional<float> relative_error_;
    optional<float> absolute_error_;
  };

  void CreateMLValue(IAllocator* alloc,
                     const std::vector<int64_t>& dims,
                     MLDataType element_type,
                     const gsl::byte* p_value,
                     size_t input_size_bytes,
                     MLValue* p_mlvalue);

  void FillFeedsAndOutputNames(const std::vector<LotusIR::NodeArg*>& input_defs,
                               const std::vector<LotusIR::NodeArg*>& output_defs,
                               std::unordered_map<std::string, MLValue>& feeds,
                               std::vector<std::string>& output_names);

  template <typename T>
  void AddData(std::vector<Data>& data, const char* name,
               const std::vector<int64_t>& dims, const T* values,
               int64_t valuesCount) {
    LOTUS_ENFORCE(TensorShape(dims).Size() == valuesCount, "Number of input values doesn't match tensor size");
    auto& allocator = AllocatorManager::Instance().GetArena(CPU);
    auto size_in_bytes = valuesCount * sizeof(T);
    void* buffer = allocator.Alloc(size_in_bytes);
    std::unique_ptr<Tensor> ptr = make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                                      TensorShape(dims),
                                                      buffer,
                                                      allocator.Info(),
                                                      &allocator);
    T* data_ptr = ptr->MutableData<T>();
    for (int64_t i = 0; i < valuesCount; i++) {
      data_ptr[i] = values[i];
    }

    MLValue value;
    value.Init(ptr.release(), DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    data.push_back({{name, &s_type_proto<T>}, value});
  }

  // Templatize the check function on type so we can compare properly (specializations defined in provider_test_utils.cc)
  template <typename T>
  void Check(const Data& output_data, const Tensor& output_tensor, size_t size);

  template <typename T>
  void Check(const Data& output_data, const T& run_output);

  const char* op_;
  const char* domain_;
  std::vector<Data> input_data_;
  std::vector<Data> output_data_;
  std::vector<std::function<void(LotusIR::Node& node)>> add_attribute_funcs_;
#if _DEBUG
  bool run_called_ = false;
#endif
};

template <typename TException>
void ExpectThrow(OpTester& test, const std::string& error_msg) {
  try {
    test.Run();
    // should throw and not reach this
    EXPECT_TRUE(false) << "Expected Run() to throw";
  } catch (TException ex) {
    EXPECT_THAT(ex.what(), testing::HasSubstr(error_msg));
  }
}

}  // namespace Test
}  // namespace Lotus

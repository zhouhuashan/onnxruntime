#pragma once

#include "core/common/logging/logging.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/inc/op_kernel_author.h"

#include "test/test_environment.h"
#include "test/framework/TestAllocatorManager.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <gsl/gsl_byte>

namespace Lotus {
namespace Test {
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

// Function templates to translate C++ types into onnx::TensorProto_DataTypes
template <typename T>
constexpr onnx::TensorProto_DataType TypeToDataType();

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<float>() { return onnx::TensorProto_DataType_FLOAT; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<double>() { return onnx::TensorProto_DataType_DOUBLE; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<int32_t>() { return onnx::TensorProto_DataType_INT32; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<int64_t>() { return onnx::TensorProto_DataType_INT64; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<bool>() { return onnx::TensorProto_DataType_BOOL; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<int8_t>() { return onnx::TensorProto_DataType_INT8; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<int16_t>() { return onnx::TensorProto_DataType_INT16; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<uint8_t>() { return onnx::TensorProto_DataType_UINT8; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<uint16_t>() { return onnx::TensorProto_DataType_UINT16; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<uint32_t>() { return onnx::TensorProto_DataType_UINT32; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<uint64_t>() { return onnx::TensorProto_DataType_UINT64; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<std::string>() { return onnx::TensorProto_DataType_STRING; }

template <>
constexpr onnx::TensorProto_DataType TypeToDataType<MLFloat16>() { return onnx::TensorProto_DataType_FLOAT16; }

template <typename T>
struct TTypeProto : onnx::TypeProto {
  TTypeProto() {
    mutable_tensor_type()->set_elem_type(TypeToDataType<T>());
  }
};

// Variable template for onnx::TensorProto_DataTypes, s_type_proto<float>, etc..
template <typename T>
const TTypeProto<T> s_type_proto;

//TypeProto for map<TKey, TVal>
template <typename TKey, typename TVal>
struct MTypeProto : onnx::TypeProto {
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
struct VectorOfMapTypeProto : onnx::TypeProto {
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
  OpTester(const char* op, int opset_version = 7, const char* domain = LotusIR::kOnnxDomain)
	  : op_(op), opset_version_(opset_version), domain_(domain) {}
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
    std::unique_ptr<std::map<TKey, TVal>> ptr = std::make_unique<std::map<TKey, TVal>>(val);
    MLValue value;
    value.Init(ptr.release(),
               DataTypeImpl::GetType<std::map<TKey, TVal>>(),
               DataTypeImpl::GetType<std::map<TKey, TVal>>()->GetDeleteFunc());
    input_data_.push_back({{name, &s_map_type_proto<TKey, TVal>}, value, optional<float>(), optional<float>()});
  }

  template <typename T>
  void AddMissingOptionalInput() {
    std::string name;  // empty == input doesn't exist
    input_data_.push_back({{name, &s_type_proto<T>}, {}, optional<float>(), optional<float>()});
  }

  template <typename T>
  void AddOutput(const char* name, const std::vector<int64_t>& dims, const std::initializer_list<T>& expected_values) {
    AddData(output_data_, name, dims, expected_values.begin(), expected_values.size());
  }

  template <typename T>
  void AddOutput(const char* name, const std::vector<int64_t>& dims, const std::vector<T>& expected_values) {
    AddData(output_data_, name, dims, expected_values.data(), expected_values.size());
  }

  template <typename T>
  void AddMissingOptionalOutput() {
    std::string name;  // empty == input doesn't exist
    output_data_.push_back({{name, &s_type_proto<T>}, {}, optional<float>(), optional<float>()});
  }

  // Add non tensor output
  template <typename TKey, typename TVal>
  void AddOutput(const char* name, const std::vector<std::map<TKey, TVal>>& val) {
    auto ptr = std::make_unique<std::vector<std::map<TKey, TVal>>>(val);
    MLValue ml_value;
    ml_value.Init(ptr.release(),
                  DataTypeImpl::GetType<std::vector<std::map<TKey, TVal>>>(),
                  DataTypeImpl::GetType<std::vector<std::map<TKey, TVal>>>()->GetDeleteFunc());
    output_data_.push_back({{name, &s_vec_map_type_proto<TKey, TVal>}, ml_value, optional<float>(), optional<float>()});
  }

  void SetOutputAbsErr(const char* name, float v);
  void SetOutputRelErr(const char* name, float v);

  template <typename T>
  void AddAttribute(std::string name, T value) {
    // Generate a the proper AddAttribute call for later
    add_attribute_funcs_.emplace_back(
        [name = std::move(name), value = std::move(value)](LotusIR::Node& node) { node.AddAttribute(name, value); });
  }

  enum class ExpectResult {
    kExpectSuccess,
    kExpectFailure
  };

  void Run(ExpectResult expect_result = ExpectResult::kExpectSuccess, const std::string& expected_failure_string = "", LotusIR::ProviderType provider_type = LotusIR::kCpuExecutionProvider);
  void RunOnCpuAndCuda(ExpectResult expect_result = ExpectResult::kExpectSuccess, const std::string& expected_failure_string = "");
  void RunOnMklDnn(ExpectResult expect_result = ExpectResult::kExpectSuccess, const std::string& expected_failure_string = "");

  struct Data {
    LotusIR::NodeArg def_;
    MLValue data_;
    optional<float> relative_error_;
    optional<float> absolute_error_;
  };

 private:
  void FillFeedsAndOutputNames(const std::vector<LotusIR::NodeArg*>& input_defs,
                               const std::vector<LotusIR::NodeArg*>& output_defs,
                               std::unordered_map<std::string, MLValue>& feeds,
                               std::vector<std::string>& output_names);

  template <typename T>
  void AddData(std::vector<Data>& data, const char* name,
               const std::vector<int64_t>& dims, const T* values,
               int64_t values_count) {
    try {
      TensorShape shape{dims};
      LOTUS_ENFORCE(shape.Size() == values_count, values_count, " input values doesn't match tensor size of ", shape.Size());

      auto allocator = Lotus::Test::AllocatorManager::Instance().GetAllocator(CPU);
      auto size_in_bytes = values_count * sizeof(T);
      void* buffer = allocator->Alloc(size_in_bytes);
      auto p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                               shape,
                                               buffer,
                                               allocator->Info(),
                                               allocator);
      auto* data_ptr = p_tensor->template MutableData<T>();
      for (int64_t i = 0; i < values_count; i++) {
        data_ptr[i] = values[i];
      }

      MLValue value;
      value.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
      data.push_back({{name, &s_type_proto<T>}, value, optional<float>(), optional<float>()});
    } catch (const std::exception& ex) {
      std::cerr << "AddData for '" << name << "' threw: " << ex.what();
      throw;
    }
  }

  const char* op_;
  const char* domain_;
  int opset_version_;
  std::vector<Data> input_data_;
  std::vector<Data> output_data_;
  std::vector<std::function<void(LotusIR::Node& node)>> add_attribute_funcs_;
#if _DEBUG
  bool run_called_{};
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

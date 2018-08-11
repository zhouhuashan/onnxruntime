#pragma once

#include <string>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "onnx/onnx_pb.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "core/common/common.h"
#include "core/common/exceptions.h"

namespace Lotus {
//maps
using MapStringToString = std::map<std::string, std::string>;
using MapStringToInt64 = std::map<std::string, int64_t>;
using MapStringToFloat = std::map<std::string, float>;
using MapStringToDouble = std::map<std::string, double>;
using MapInt64ToString = std::map<int64_t, std::string>;
using MapInt64ToInt64 = std::map<int64_t, int64_t>;
using MapInt64ToFloat = std::map<int64_t, float>;
using MapInt64ToDouble = std::map<int64_t, double>;

//vectors/sequences
using VectorString = std::vector<std::string>;
using VectorInt64 = std::vector<int64_t>;
using VectorFloat = std::vector<float>;
using VectorDouble = std::vector<double>;
using VectorMapStringToFloat = std::vector<MapStringToFloat>;
using VectorMapInt64ToFloat = std::vector<MapInt64ToFloat>;

class DataTypeImpl;
class TensorTypeBase;
// DataTypeImpl pointer as unique DataTypeImpl identifier.
using MLDataType = const DataTypeImpl*;
using DeleteFunc = std::function<void(void*)>;
using CreateFunc = std::function<void*()>;

template <typename T>
static void Delete(void* p) {
  delete static_cast<T*>(p);
}

class DataTypeImpl {
 public:
  virtual ~DataTypeImpl() = default;

  // TODO: this API will be used to check type in runtime really
  // matches type defined in a model.
  // 1) This should be overriden by sub-classes and have detail check implementation.
  // 2) The reason is checking compatibility is because one runtime type may be
  // able to represent multiple type protos, for example, "float" could match float16, float.
  // 3) After sub-class having the implementation of this function in-place, we should either
  // change the return value from true to false here or make this function as a pure virtual function.
  virtual bool IsCompatible(const onnx::TypeProto& type_proto) const = 0;
  virtual size_t Size() const = 0;

  virtual DeleteFunc GetDeleteFunc() const = 0;

  virtual bool IsTensorType() const {
    return false;
  }

  // Returns this if this is of tensor-type and null otherwise
  virtual const TensorTypeBase* AsTensorType() const {
    return nullptr;
  }

  // Return the type meta that we are using in the runtime.
  template <typename T>
  static MLDataType GetType();

  // Return the types for a concrete tensor type, like Tensor_Float
  template <typename T>
  static MLDataType GetTensorType();

  static MLDataType TypeFromProto(const onnx::TypeProto& proto);

  static const std::vector<MLDataType>& AllTensorTypes();
  static const std::vector<MLDataType>& AllFixedSizeTensorTypes();
};

std::ostream& operator<<(std::ostream& out, MLDataType data_type);

class TensorTypeBase : public DataTypeImpl {
 public:
  static MLDataType Type() {
    static TensorTypeBase tensor_base;
    return &tensor_base;
  }

  bool IsTensorType() const override {
    return true;
  }

  const TensorTypeBase* AsTensorType() const override {
    return this;
  }

  size_t Size() const override;

  DeleteFunc GetDeleteFunc() const override;

  virtual MLDataType GetElementType() const {
    // should never reach here.
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
  bool IsCompatible(const onnx::TypeProto& /*type_proto*/) const override {
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

 protected:
  TensorTypeBase() = default;
};

template <typename elemT>
struct TensorType : public TensorTypeBase {
  static MLDataType Type() {
    static TensorType tensor_type;
    return &tensor_type;
  }

  MLDataType GetElementType() const override {
    return DataTypeImpl::GetType<elemT>();
  }
  bool IsCompatible(const onnx::TypeProto& type_proto) const override;

 private:
  TensorType() = default;
};

class NonTensorTypeBase : public DataTypeImpl {
 public:
  size_t Size() const override = 0;

  DeleteFunc GetDeleteFunc() const override = 0;

  virtual CreateFunc GetCreateFunc() const = 0;

 protected:
  NonTensorTypeBase() = default;
};

template <typename T>
class NonTensorType : public NonTensorTypeBase {
 public:
  static MLDataType Type() {
    static NonTensorType non_tensor_type;
    return &non_tensor_type;
  }

  CreateFunc GetCreateFunc() const override {
    return []() { return new T(); };
  }

  size_t Size() const override {
    return sizeof(T);
  }

  DeleteFunc GetDeleteFunc() const override {
    return &Delete<T>;
  }

  bool IsCompatible(const onnx::TypeProto& type_proto) const override;

 private:
  NonTensorType() = default;
};

template <typename T>
class NonOnnxType : public DataTypeImpl {
 public:
  bool IsCompatible(const onnx::TypeProto&) const override {
    return false;
  }

  static MLDataType Type() {
    static NonOnnxType non_tensor_type;
    return &non_tensor_type;
  }

  size_t Size() const override {
    return sizeof(T);
  }

  DeleteFunc GetDeleteFunc() const override {
    return &Delete<T>;
  }

 private:
  NonOnnxType() = default;
};

#define LOTUS_REGISTER_NON_TENSOR_TYPE(TYPE, COMPATIBLE_CONDITION)            \
  template <>                                                                 \
  MLDataType DataTypeImpl::GetType<TYPE>() {                                  \
    return NonTensorType<TYPE>::Type();                                       \
  }                                                                           \
  template <>                                                                 \
  bool NonTensorType<TYPE>::IsCompatible(const TypeProto& type_proto) const { \
    return COMPATIBLE_CONDITION;                                              \
  }

#define LOTUS_REGISTER_MAP(TYPE, KEY, Value)                                                                                 \
  LOTUS_REGISTER_NON_TENSOR_TYPE(TYPE,                                                                                       \
                                 type_proto.value_case() == TypeProto::ValueCase::kMapType &&                                \
                                     type_proto.map_type().key_type() == KEY &&                                              \
                                     type_proto.map_type().value_type().value_case() == TypeProto::ValueCase::kTensorType && \
                                     type_proto.map_type().value_type().tensor_type().shape().dim_size() == 0 &&             \
                                     type_proto.map_type().value_type().tensor_type().has_elem_type() &&                     \
                                     type_proto.map_type().value_type().tensor_type().elem_type() == Value);

#define LOTUS_REGISTER_SEQ(TYPE, Value)                                                                                          \
  LOTUS_REGISTER_NON_TENSOR_TYPE(TYPE,                                                                                           \
                                 type_proto.value_case() == TypeProto::ValueCase::kSequenceType &&                               \
                                     type_proto.sequence_type().elem_type().value_case() == TypeProto::ValueCase::kTensorType && \
                                     type_proto.sequence_type().elem_type().tensor_type().shape().dim_size() == 0 &&             \
                                     type_proto.sequence_type().elem_type().tensor_type().has_elem_type() &&                     \
                                     type_proto.sequence_type().elem_type().tensor_type().elem_type() == TensorProto_DataType_STRING);

#define LOTUS_REGISTER_TENSOR_TYPE(ELEM_TYPE)           \
  template <>                                           \
  MLDataType DataTypeImpl::GetTensorType<ELEM_TYPE>() { \
    return TensorType<ELEM_TYPE>::Type();               \
  }

#define LOTUS_REGISTER_NON_ONNX_TYPE(TYPE)   \
  template <>                                \
  MLDataType DataTypeImpl::GetType<TYPE>() { \
    return NonOnnxType<TYPE>::Type();        \
  }
}  // namespace Lotus

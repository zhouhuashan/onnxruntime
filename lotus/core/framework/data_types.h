#pragma once

#include <string>
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "onnx/onnx_pb.h"

using namespace onnx;

namespace Lotus {
class DataTypeImpl;
class TensorTypeBase;
// DataTypeImpl pointer as unique DataTypeImpl identifier.
typedef const DataTypeImpl* MLDataType;
typedef std::function<void(void*)> DeleteFunc;
typedef std::function<void*(void)> CreateFunc;

template <typename T>
static void Delete(void* p) {
  delete static_cast<T*>(p);
}

class DataTypeImpl {
 public:
  virtual ~DataTypeImpl() {}

  virtual bool IsCompatible(const TypeProto& type_proto) const {
    // TODO: this API will be used to check type in runtime really
    // matches type defined in a model.
    // 1) This should be overriden by sub-classes and have detail check implementation.
    // 2) The reason is checking compatibility is because one runtime type may be
    // able to represent multiple type protos, for example, "float" could match float16, float.
    // 3) After sub-class having the implementation of this function in-place, we should either
    // change the return value from true to false here or make this function as a pure virtual function.
    UNUSED_PARAMETER(type_proto);
    return true;
  }

  virtual const size_t Size() const = 0;

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
  static MLDataType ElementTypeFromProto(onnx::TensorProto_DataType type);
};

std::ostream& operator<<(std::ostream& out, const MLDataType data_type);

class TensorTypeBase : public DataTypeImpl {
 public:
  static MLDataType Type() {
    static TensorTypeBase tensor_base;
    return &tensor_base;
  }

  virtual bool IsTensorType() const override {
    return true;
  }

  virtual const TensorTypeBase* AsTensorType() const override {
    return this;
  }

  virtual const size_t Size() const;

  virtual DeleteFunc GetDeleteFunc() const;

  virtual MLDataType GetElementType() const {
    // should never reach here.
    LOTUS_NOT_IMPLEMENTED;
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

  virtual MLDataType GetElementType() const {
    return DataTypeImpl::GetType<elemT>();
  }

 private:
  TensorType() = default;
};

class NonTensorTypeBase : public DataTypeImpl {
 public:
  virtual const size_t Size() const = 0;

  virtual DeleteFunc GetDeleteFunc() const = 0;

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

  CreateFunc GetCreateFunc() const {
    return []() { return new T(); };
  }

  virtual const size_t Size() const override {
    return sizeof(T);
  }

  virtual DeleteFunc GetDeleteFunc() const override {
    return &Delete<T>;
  }

 private:
  NonTensorType() = default;
};

template <typename T>
class NonOnnxType : public DataTypeImpl {
 public:
  virtual bool IsCompatible(const TypeProto& type_proto) const {
    UNUSED_PARAMETER(type_proto);
    return false;
  }

  static MLDataType Type() {
    static NonOnnxType non_tensor_type;
    return &non_tensor_type;
  }

  virtual const size_t Size() const override {
    return sizeof(T);
  }

  virtual DeleteFunc GetDeleteFunc() const override {
    return &Delete<T>;
  }

 private:
  NonOnnxType() = default;
};

#define LOTUS_REGISTER_NON_TENSOR_TYPE(TYPE) \
  template <>                                \
  MLDataType DataTypeImpl::GetType<TYPE>() { \
    return NonTensorType<TYPE>::Type();      \
  }

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

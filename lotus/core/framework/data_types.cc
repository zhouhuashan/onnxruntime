#include "core/framework/data_types.h"
#include "core/framework/tensor.h"

namespace Lotus {
template <>
MLDataType DataTypeImpl::GetType<Tensor>() {
  return TensorTypeBase::Type();
}

const size_t TensorTypeBase::Size() const {
  return sizeof(Tensor);
}

DeleteFunc TensorTypeBase::GetDeleteFunc() const {
  return &Delete<Tensor>;
}

LOTUS_REGISTER_TENSOR_TYPE(int);
LOTUS_REGISTER_TENSOR_TYPE(float);
LOTUS_REGISTER_TENSOR_TYPE(bool);
LOTUS_REGISTER_TENSOR_TYPE(std::string);
LOTUS_REGISTER_TENSOR_TYPE(uint8_t);
LOTUS_REGISTER_TENSOR_TYPE(uint16_t);
LOTUS_REGISTER_TENSOR_TYPE(int16_t);
LOTUS_REGISTER_TENSOR_TYPE(int64_t);
LOTUS_REGISTER_TENSOR_TYPE(double);
LOTUS_REGISTER_TENSOR_TYPE(uint32_t);
LOTUS_REGISTER_TENSOR_TYPE(uint64_t);

MLDataType DataTypeImpl::TypeFromProto(const onnx::TypeProto& proto) {
  switch (proto.value_case()) {
    case TypeProto::ValueCase::kTensorType: {
      auto tensor_type = proto.tensor_type();
      LOTUS_ENFORCE(tensor_type.has_elem_type());
      switch (tensor_type.elem_type()) {
        case TensorProto_DataType_FLOAT:
          return DataTypeImpl::GetTensorType<float>();
        case TensorProto_DataType_BOOL:
          return DataTypeImpl::GetTensorType<bool>();
        case TensorProto_DataType_INT32:
          return DataTypeImpl::GetTensorType<int>();
        case TensorProto_DataType_DOUBLE:
          return DataTypeImpl::GetTensorType<double>();
        case TensorProto_DataType_STRING:
          return DataTypeImpl::GetTensorType<std::string>();
        case TensorProto_DataType_UINT8:
          return DataTypeImpl::GetTensorType<uint8_t>();
        case TensorProto_DataType_UINT16:
          return DataTypeImpl::GetTensorType<uint16_t>();
        case TensorProto_DataType_INT16:
          return DataTypeImpl::GetTensorType<int16_t>();
        case TensorProto_DataType_INT64:
          return DataTypeImpl::GetTensorType<int64_t>();
        case TensorProto_DataType_UINT32:
          return DataTypeImpl::GetTensorType<uint32_t>();
        case TensorProto_DataType_UINT64:
          return DataTypeImpl::GetTensorType<uint64_t>();
        default:
          LOTUS_NOT_IMPLEMENTED;
      }
    } break;
    default:
      LOTUS_NOT_IMPLEMENTED;
  }
}

//Below are the types the we need to execute the runtime
//They are not compatible with TypeProto in ONNX.
LOTUS_REGISTER_NON_ONNX_TYPE(int);
LOTUS_REGISTER_NON_ONNX_TYPE(float);
LOTUS_REGISTER_NON_ONNX_TYPE(bool);
LOTUS_REGISTER_NON_ONNX_TYPE(std::string);
LOTUS_REGISTER_NON_ONNX_TYPE(uint8_t);
LOTUS_REGISTER_NON_ONNX_TYPE(uint16_t);
LOTUS_REGISTER_NON_ONNX_TYPE(int16_t);
LOTUS_REGISTER_NON_ONNX_TYPE(int64_t);
LOTUS_REGISTER_NON_ONNX_TYPE(double);
LOTUS_REGISTER_NON_ONNX_TYPE(uint32_t);
LOTUS_REGISTER_NON_ONNX_TYPE(uint64_t);
}  // namespace Lotus

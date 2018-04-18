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
LOTUS_REGISTER_TENSOR_TYPE(int8_t);
LOTUS_REGISTER_TENSOR_TYPE(uint8_t);
LOTUS_REGISTER_TENSOR_TYPE(uint16_t);
LOTUS_REGISTER_TENSOR_TYPE(int16_t);
LOTUS_REGISTER_TENSOR_TYPE(int64_t);
LOTUS_REGISTER_TENSOR_TYPE(double);
LOTUS_REGISTER_TENSOR_TYPE(uint32_t);
LOTUS_REGISTER_TENSOR_TYPE(uint64_t);

//maps
using MapStringToString = std::map<std::string, std::string>;
using MapStringToInt64 = std::map<std::string, int64_t>;
using MapStringFloat = std::map<std::string, float>;
using MapStringDouble = std::map<std::string, double>;
using MapInt64ToString = std::map<int64_t, std::string>;
using MapInt64ToInt64 = std::map<int64_t, int64_t>;
using MapInt64Float = std::map<int64_t, float>;
using MapInt64Double = std::map<int64_t, double>;

//vectors/sequences
using VectorString = std::vector<std::string>;
using VectorInt64 = std::vector<int64_t>;
using VectorFloat = std::vector<float>;
using VectorDouble = std::vector<double>;

LOTUS_REGISTER_NON_TENSOR_TYPE(MapStringToString);
LOTUS_REGISTER_NON_TENSOR_TYPE(MapStringToInt64);
LOTUS_REGISTER_NON_TENSOR_TYPE(MapStringFloat);
LOTUS_REGISTER_NON_TENSOR_TYPE(MapStringDouble);
LOTUS_REGISTER_NON_TENSOR_TYPE(MapInt64ToString);
LOTUS_REGISTER_NON_TENSOR_TYPE(MapInt64ToInt64);
LOTUS_REGISTER_NON_TENSOR_TYPE(MapInt64Float);
LOTUS_REGISTER_NON_TENSOR_TYPE(MapInt64Double);
LOTUS_REGISTER_NON_TENSOR_TYPE(VectorString);
LOTUS_REGISTER_NON_TENSOR_TYPE(VectorFloat);
LOTUS_REGISTER_NON_TENSOR_TYPE(VectorInt64);
LOTUS_REGISTER_NON_TENSOR_TYPE(VectorDouble);

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
    case TypeProto::ValueCase::kMapType: {
      auto maptype = proto.map_type();
      auto keytype = maptype.key_type();
      auto value_type = maptype.value_type();
      if (value_type.value_case() != TypeProto::ValueCase::kTensorType ||
          value_type.tensor_type().shape().dim_size() != 0) {
        LOTUS_THROW("Nested map/sequence type is not supported");
      }

      auto value_elem_type = value_type.tensor_type().elem_type();
      switch (value_elem_type) {
        case TensorProto_DataType_STRING: {
          switch (keytype) {
            case TensorProto_DataType_STRING:
              return DataTypeImpl::GetType<MapStringToInt64>();
            case TensorProto_DataType_INT64:
              return DataTypeImpl::GetType<MapInt64ToString>();
            default:
              LOTUS_THROW("Map with key type: ", keytype, " is not supported");
          }
        }
        case TensorProto_DataType_INT64:
          switch (keytype) {
            case TensorProto_DataType_STRING:
              return DataTypeImpl::GetType<MapStringToString>();
            case TensorProto_DataType_INT64:
              return DataTypeImpl::GetType<MapInt64ToInt64>();
            default:
              LOTUS_THROW("Map with key type: ", keytype, " is not supported");
          }
        case TensorProto_DataType_FLOAT:
          switch (keytype) {
            case TensorProto_DataType_STRING:
              return DataTypeImpl::GetType<MapStringFloat>();
            case TensorProto_DataType_INT64:
              return DataTypeImpl::GetType<MapInt64Float>();
            default:
              LOTUS_THROW("Map with key type: ", keytype, " is not supported");
          }
        case TensorProto_DataType_DOUBLE:
          switch (keytype) {
            case TensorProto_DataType_STRING:
              return DataTypeImpl::GetType<MapStringDouble>();
            case TensorProto_DataType_INT64:
              return DataTypeImpl::GetType<MapInt64Double>();
            default:
              LOTUS_THROW("Map with key type: ", keytype, " is not supported");
          }
        default:
          LOTUS_THROW("Map with value type: ", value_elem_type, " is not supported");
      }
    } break;
    case TypeProto::ValueCase::kSequenceType: {
      auto seq_type = proto.sequence_type();
      auto val_type = seq_type.elem_type();
      if (val_type.value_case() != TypeProto::ValueCase::kTensorType ||
          val_type.tensor_type().shape().dim_size() != 0) {
        LOTUS_THROW("Nested map/sequence type is not supported");
      }
      auto val_elem_type = val_type.tensor_type().elem_type();
      switch (val_elem_type) {
        case TensorProto_DataType_STRING:
          return DataTypeImpl::GetType<VectorString>();
        case TensorProto_DataType_INT64:
          return DataTypeImpl::GetType<VectorInt64>();
        case TensorProto_DataType_FLOAT:
          return DataTypeImpl::GetType<VectorFloat>();
        case TensorProto_DataType_DOUBLE:
          return DataTypeImpl::GetType<VectorDouble>();
        default:
          LOTUS_THROW("Sequence with value type: ", val_elem_type, " is not supported");
      }
    }
    default:
      LOTUS_THROW("Onnx type: ", proto.value_case(), " is not supported.");
  }
}

//Below are the types the we need to execute the runtime
//They are not compatible with TypeProto in ONNX.
LOTUS_REGISTER_NON_ONNX_TYPE(int);
LOTUS_REGISTER_NON_ONNX_TYPE(float);
LOTUS_REGISTER_NON_ONNX_TYPE(bool);
LOTUS_REGISTER_NON_ONNX_TYPE(std::string);
LOTUS_REGISTER_NON_ONNX_TYPE(int8_t);
LOTUS_REGISTER_NON_ONNX_TYPE(uint8_t);
LOTUS_REGISTER_NON_ONNX_TYPE(uint16_t);
LOTUS_REGISTER_NON_ONNX_TYPE(int16_t);
LOTUS_REGISTER_NON_ONNX_TYPE(int64_t);
LOTUS_REGISTER_NON_ONNX_TYPE(double);
LOTUS_REGISTER_NON_ONNX_TYPE(uint32_t);
LOTUS_REGISTER_NON_ONNX_TYPE(uint64_t);

MLDataType DataTypeImpl::ElementTypeFromProto(onnx::TensorProto_DataType type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return DataTypeImpl::GetType<float>();
    case TensorProto_DataType_BOOL:
      return DataTypeImpl::GetType<bool>();
    case TensorProto_DataType_INT32:
      return DataTypeImpl::GetType<int>();
    case TensorProto_DataType_DOUBLE:
      return DataTypeImpl::GetType<double>();
    case TensorProto_DataType_STRING:
      return DataTypeImpl::GetType<std::string>();
    case TensorProto_DataType_UINT8:
      return DataTypeImpl::GetType<uint8_t>();
    case TensorProto_DataType_UINT16:
      return DataTypeImpl::GetType<uint16_t>();
    case TensorProto_DataType_INT16:
      return DataTypeImpl::GetType<int16_t>();
    case TensorProto_DataType_INT64:
      return DataTypeImpl::GetType<int64_t>();
    case TensorProto_DataType_UINT32:
      return DataTypeImpl::GetType<uint32_t>();
    case TensorProto_DataType_UINT64:
      return DataTypeImpl::GetType<uint64_t>();
    default:
      LOTUS_NOT_IMPLEMENTED;
  }
}
}  // namespace Lotus

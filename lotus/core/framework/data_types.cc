#include "core/framework/data_types.h"
#include "core/framework/tensor.h"
#include "onnx/defs/data_type_utils.h"

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

template <>
bool TensorType<float>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_FLOAT;
}

template <>
bool TensorType<double>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_DOUBLE;
}

template <>
bool TensorType<std::string>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_STRING;
}

template <>
bool TensorType<int32_t>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_INT32;
}

template <>
bool TensorType<bool>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_BOOL;
}

template <>
bool TensorType<int8_t>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_INT8;
}

template <>
bool TensorType<int16_t>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_INT16;
}

template <>
bool TensorType<uint8_t>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_UINT8;
}

template <>
bool TensorType<uint16_t>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_UINT16;
}

template <>
bool TensorType<uint32_t>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_UINT32;
}

template <>
bool TensorType<int64_t>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_INT64;
}

template <>
bool TensorType<uint64_t>::IsCompatible(const TypeProto& type_proto) const {
  return type_proto.value_case() == TypeProto::ValueCase::kTensorType && type_proto.tensor_type().has_elem_type() && type_proto.tensor_type().elem_type() == TensorProto_DataType_UINT64;
}

LOTUS_REGISTER_TENSOR_TYPE(int32_t);
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

LOTUS_REGISTER_MAP(MapStringToString, TensorProto_DataType_STRING, TensorProto_DataType_STRING);
LOTUS_REGISTER_MAP(MapStringToInt64, TensorProto_DataType_STRING, TensorProto_DataType_INT64);
LOTUS_REGISTER_MAP(MapStringToFloat, TensorProto_DataType_STRING, TensorProto_DataType_FLOAT);
LOTUS_REGISTER_MAP(MapStringToDouble, TensorProto_DataType_STRING, TensorProto_DataType_DOUBLE);
LOTUS_REGISTER_MAP(MapInt64ToString, TensorProto_DataType_INT64, TensorProto_DataType_STRING);
LOTUS_REGISTER_MAP(MapInt64ToInt64, TensorProto_DataType_INT64, TensorProto_DataType_INT64);
LOTUS_REGISTER_MAP(MapInt64ToFloat, TensorProto_DataType_INT64, TensorProto_DataType_FLOAT);
LOTUS_REGISTER_MAP(MapInt64ToDouble, TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE);
LOTUS_REGISTER_SEQ(VectorString, TensorProto_DataType_STRING);
LOTUS_REGISTER_SEQ(VectorFloat, TensorProto_DataType_FLOAT);
LOTUS_REGISTER_SEQ(VectorInt64, TensorProto_DataType_INT64);
LOTUS_REGISTER_SEQ(VectorDouble, TensorProto_DataType_DOUBLE);
LOTUS_REGISTER_NON_TENSOR_TYPE(VectorMapStringToFloat,
                               type_proto.value_case() == TypeProto::ValueCase::kSequenceType &&
                                   type_proto.sequence_type().elem_type().value_case() == TypeProto::ValueCase::kMapType &&
                                   type_proto.sequence_type().elem_type().map_type().key_type() == TensorProto_DataType_STRING &&
                                   type_proto.sequence_type().elem_type().map_type().value_type().value_case() == TypeProto::ValueCase::kTensorType &&
                                   type_proto.sequence_type().elem_type().map_type().value_type().tensor_type().shape().dim_size() == 0 &&
                                   type_proto.sequence_type().elem_type().map_type().value_type().tensor_type().has_elem_type() &&
                                   type_proto.sequence_type().elem_type().map_type().value_type().tensor_type().elem_type() == TensorProto_DataType_FLOAT);
LOTUS_REGISTER_NON_TENSOR_TYPE(VectorMapInt64ToFloat,
                               type_proto.value_case() == TypeProto::ValueCase::kSequenceType &&
                                   type_proto.sequence_type().elem_type().value_case() == TypeProto::ValueCase::kMapType &&
                                   type_proto.sequence_type().elem_type().map_type().key_type() == TensorProto_DataType_INT64 &&
                                   type_proto.sequence_type().elem_type().map_type().value_type().value_case() == TypeProto::ValueCase::kTensorType &&
                                   type_proto.sequence_type().elem_type().map_type().value_type().tensor_type().shape().dim_size() == 0 &&
                                   type_proto.sequence_type().elem_type().map_type().value_type().tensor_type().has_elem_type() &&
                                   type_proto.sequence_type().elem_type().map_type().value_type().tensor_type().elem_type() == TensorProto_DataType_FLOAT);

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
          return DataTypeImpl::GetTensorType<int32_t>();
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
          LOTUS_NOT_IMPLEMENTED("tensor type ", tensor_type.elem_type(), " is not supported");
      }
    } break;
    case TypeProto::ValueCase::kMapType: {
      auto maptype = proto.map_type();
      auto keytype = maptype.key_type();
      auto value_type = maptype.value_type();
      if (value_type.value_case() != TypeProto::ValueCase::kTensorType ||
          value_type.tensor_type().shape().dim_size() != 0) {
        LOTUS_NOT_IMPLEMENTED("Nested map/sequence type is not supported");
      }

      auto value_elem_type = value_type.tensor_type().elem_type();
      switch (value_elem_type) {
        case TensorProto_DataType_STRING: {
          switch (keytype) {
            case TensorProto_DataType_STRING:
              return DataTypeImpl::GetType<MapStringToString>();
            case TensorProto_DataType_INT64:
              return DataTypeImpl::GetType<MapInt64ToString>();
            default:
              LOTUS_NOT_IMPLEMENTED("Map with key type: ", keytype, " is not supported");
          }
        }
        case TensorProto_DataType_INT64:
          switch (keytype) {
            case TensorProto_DataType_STRING:
              return DataTypeImpl::GetType<MapStringToInt64>();
            case TensorProto_DataType_INT64:
              return DataTypeImpl::GetType<MapInt64ToInt64>();
            default:
              LOTUS_NOT_IMPLEMENTED("Map with key type: ", keytype, " is not supported");
          }
        case TensorProto_DataType_FLOAT:
          switch (keytype) {
            case TensorProto_DataType_STRING:
              return DataTypeImpl::GetType<MapStringToFloat>();
            case TensorProto_DataType_INT64:
              return DataTypeImpl::GetType<MapInt64ToFloat>();
            default:
              LOTUS_NOT_IMPLEMENTED("Map with key type: ", keytype, " is not supported");
          }
        case TensorProto_DataType_DOUBLE:
          switch (keytype) {
            case TensorProto_DataType_STRING:
              return DataTypeImpl::GetType<MapStringToDouble>();
            case TensorProto_DataType_INT64:
              return DataTypeImpl::GetType<MapInt64ToDouble>();
            default:
              LOTUS_NOT_IMPLEMENTED("Map with key type: ", keytype, " is not supported");
          }
        default:
          LOTUS_NOT_IMPLEMENTED("Map with value type: ", value_elem_type, " is not supported");
      }
    } break;
    case TypeProto::ValueCase::kSequenceType: {
      auto& seq_type = proto.sequence_type();
      auto& val_type = seq_type.elem_type();

      switch (val_type.value_case()) {
        case TypeProto::ValueCase::kMapType: {
          auto& maptype = val_type.map_type();
          auto keytype = maptype.key_type();
          auto& value_type = maptype.value_type();
          if (value_type.value_case() != TypeProto::ValueCase::kTensorType ||
              value_type.tensor_type().shape().dim_size() != 0) {
            LOTUS_THROW("Nested map/sequence type is not supported");
          }

          auto value_elem_type = value_type.tensor_type().elem_type();
          switch (value_elem_type) {
            case TensorProto_DataType_FLOAT: {
              switch (keytype) {
                case TensorProto_DataType_STRING:
                  return DataTypeImpl::GetType<VectorMapStringToFloat>();
                case TensorProto_DataType_INT64:
                  return DataTypeImpl::GetType<VectorMapInt64ToFloat>();
                default:
                  LOTUS_THROW("Map with key type: ", keytype, " is not supported");
              }
            }
            default:
              LOTUS_THROW("Sequence type that has a map of value type other than float not supported for now.");
          }
        }
        case TypeProto::ValueCase::kTensorType: {
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
      }
    }
    default:
      throw Lotus::NotImplementedException(Lotus::MakeString("Onnx type: ", proto.value_case(), " is not supported."));
  }
}

//Below are the types the we need to execute the runtime
//They are not compatible with TypeProto in ONNX.
LOTUS_REGISTER_NON_ONNX_TYPE(int32_t);
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
      LOTUS_NOT_IMPLEMENTED(__FUNCTION__, ":tensor type ", type, " is not supported");
  }
}

// helper to stream. expected to only be used for error output, so any typeid lookup
// cost should be fine. alternative would be to add a static string field to DataTypeImpl
// that we set in the register macro to the type name, and output that instead.
std::ostream& operator<<(std::ostream& out, const MLDataType data_type) {
  return out << typeid(*data_type).name();
}

}  // namespace Lotus

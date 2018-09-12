#include "mldata_type_utils.h"

namespace onnxruntime {
namespace Utils {
MLDataType GetMLDataType(const onnxruntime::NodeArg& arg) {
  const ONNX_NAMESPACE::DataType ptype = arg.Type();
  const ONNX_NAMESPACE::TypeProto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(ptype);
  return DataTypeImpl::TypeFromProto(type_proto);
}
}  // namespace Utils
}  // namespace onnxruntime

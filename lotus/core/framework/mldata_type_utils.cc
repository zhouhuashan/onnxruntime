#include "mldata_type_utils.h"

namespace Lotus {
namespace Utils {
MLDataType GetMLDataType(const LotusIR::NodeArg& arg) {
  const ONNX_NAMESPACE::DataType ptype = arg.Type();
  const ONNX_NAMESPACE::TypeProto& type_proto = ONNX_NAMESPACE::Utils::DataTypeUtils::ToTypeProto(ptype);
  return DataTypeImpl::TypeFromProto(type_proto);
}
}  // namespace Utils
}  // namespace Lotus

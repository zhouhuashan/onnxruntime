#include "core/providers/cpu/generator/constant.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/tensorutils.h"

namespace Lotus {
Status Constant::Compute(OpKernelContext* context) const {
  TensorShape shape(Utils::GetTensorShapeFromTensorProto(value_));

  auto& C = *context->Output(0, shape);
  switch (value_.data_type()) {
    case TensorProto_DataType_FLOAT:
      return Lotus::Utils::TensorUtils::UnpackTensor(value_, C.MutableData<float>(), shape.Size());
    case TensorProto_DataType_DOUBLE:
      return Lotus::Utils::TensorUtils::UnpackTensor(value_, C.MutableData<double>(), shape.Size());
    case TensorProto_DataType_INT64:
      return Lotus::Utils::TensorUtils::UnpackTensor(value_, C.MutableData<int64_t>(), shape.Size());
    default:
      std::ostringstream oss;
      oss << "data type of " << value_.data_type() << " is not supported by Constant OP";
      return Lotus::Common::Status(StatusCategory::LOTUS, StatusCode::NOT_IMPLEMENTED, oss.str());
  }
}

REGISTER_KERNEL(KernelDefBuilder("Constant")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>(), DataTypeImpl::GetTensorType<int64_t>()}),
                Constant);

}  // namespace Lotus

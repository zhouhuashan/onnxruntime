#include "core/providers/cpu/tensor/shape_op.h"

namespace Lotus {

const std::vector<MLDataType> shapeOpTypeConstraints{
    DataTypeImpl::GetTensorType<bool>(),
    DataTypeImpl::GetTensorType<float>(),
    DataTypeImpl::GetTensorType<double>(),
    DataTypeImpl::GetTensorType<int16_t>(),
    DataTypeImpl::GetTensorType<int>(),
    DataTypeImpl::GetTensorType<int64_t>(),
    DataTypeImpl::GetTensorType<uint8_t>(),
    DataTypeImpl::GetTensorType<uint16_t>()};

REGISTER_KERNEL(KernelDefBuilder("Shape")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", shapeOpTypeConstraints)
                    .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
                Shape<float>);
}  // namespace Lotus

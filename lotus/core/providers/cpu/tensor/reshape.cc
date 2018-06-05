#include "core/providers/cpu/tensor/reshape.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("Reshape")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(5)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .Alias(0, 0)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
                Reshape<float>);

REGISTER_KERNEL(KernelDefBuilder("Reshape")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 4)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .Alias(0, 0)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Reshape_1<float>);

}  // namespace Lotus

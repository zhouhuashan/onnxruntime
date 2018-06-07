#include "core/providers/cpu/tensor/gather.h"

namespace Lotus {
REGISTER_KERNEL(KernelDefBuilder("Gather")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int64_t>()),
                Gather<float, int64_t>);

REGISTER_KERNEL(KernelDefBuilder("Gather")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int32_t>()),
                Gather<float, int32_t>);
}  // namespace Lotus

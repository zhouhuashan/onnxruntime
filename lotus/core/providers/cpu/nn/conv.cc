#include "core/providers/cpu/nn/conv.h"

namespace Lotus {
REGISTER_KERNEL(KernelDefBuilder("Conv")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("X", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("W", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("B", DataTypeImpl::GetTensorType<float>()),
                Conv<float>);
}

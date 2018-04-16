#include "core/providers/cpu/nn/conv_transpose.h"

namespace Lotus {
REGISTER_KERNEL(KernelDefBuilder("ConvTranspose")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ConvTranspose<float>);
}

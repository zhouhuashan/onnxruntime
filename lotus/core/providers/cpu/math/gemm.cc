#include "core/providers/cpu/math/gemm.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("Gemm")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(7)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Gemm<float, float, float, float>);
}

#include "core/providers/cpu/math/gemm.h"

namespace Lotus {

REGISTER_KERNEL(KernelDef("Gemm")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Gemm<float, float, float, float>);
}

#include "core/providers/cpu/math/gemm.h"

namespace Lotus {

REGISTER_KERNEL(KernelDef("Gemm")
                    .Domain(LotusIR::c_onnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::c_cpuExecutionProvider)
                    .TypeConstraint("X", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("W", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("B", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("B", DataTypeImpl::GetTensorType<float>()),
                Gemm<float, float, float, float>);
}

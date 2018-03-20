#include "core/providers/cpu/tensor/reshape.h"

namespace Lotus {

//non-copying reshape kernel
REGISTER_KERNEL(KernelDef("Reshape")
    .Domain(LotusIR::c_onnxDomain)
    .SinceVersion(1, 2)
    .Provider(LotusIR::c_cpuExecutionProvider)
    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
    .Alias(0, 0),
    Reshape<float>);

//copying reshape kernel
REGISTER_KERNEL(KernelDef("Reshape")
    .Domain(LotusIR::c_onnxDomain)
    .SinceVersion(1, 2)
    .Provider(LotusIR::c_cpuExecutionProvider)
    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Reshape<float>);

}
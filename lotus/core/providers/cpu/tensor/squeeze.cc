#include "core/providers/cpu/tensor/squeeze.h"

namespace Lotus {

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Squeeze,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .Alias(0, 0),
    Squeeze<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Squeeze,
    1,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())
        .Alias(0, 0),
    Squeeze<int32_t>);

}  // namespace Lotus

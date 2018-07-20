#include "core/providers/cpu/nn/conv_impl.h"

namespace Lotus {
ONNX_CPU_OPERATOR_KERNEL(
    Conv,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);
}

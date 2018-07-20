#include "core/providers/cpu/nn/conv_transpose.h"

namespace Lotus {
ONNX_CPU_OPERATOR_KERNEL(
    ConvTranspose,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ConvTranspose<float>);
}

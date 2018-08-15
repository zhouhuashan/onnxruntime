#include "core/providers/cpu/tensor/squeeze.h"

namespace Lotus {

ONNX_CPU_OPERATOR_KERNEL(
    Squeeze,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

}  // namespace Lotus

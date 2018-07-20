#include "core/providers/cpu/tensor/squeeze.h"

namespace Lotus {

ONNX_CPU_OPERATOR_KERNEL(
    Squeeze,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    Squeeze<float>);

}

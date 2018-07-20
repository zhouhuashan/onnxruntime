#include "core/providers/cpu/tensor/mean_variance_normalization.h"

namespace Lotus {
ONNX_CPU_OPERATOR_KERNEL(
    MeanVarianceNormalization,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MeanVarianceNormalization<float>);
}  // namespace Lotus

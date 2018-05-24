#include "core/providers/cpu/tensor/mean_variance_normalization.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("MeanVarianceNormalization")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                MeanVarianceNormalization<float>);
}  // namespace Lotus

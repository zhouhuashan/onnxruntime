#include "core/providers/cpu/tensor/reshape.h"

namespace Lotus {

//copying reshape kernel
REGISTER_KERNEL(KernelDefBuilder("Reshape")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Reshape<float>);

}  // namespace Lotus

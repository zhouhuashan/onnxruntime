#include "reshape.h"

namespace Lotus {
namespace Cuda {
REGISTER_KERNEL(KernelDefBuilder("Reshape")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(5)
                    .Provider(LotusIR::kCudaExecutionProvider)
                    .Alias(0, 0)
                    .InputMemoryType<kMemTypeCPUInput>(1)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                    .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
                Reshape<float>);

REGISTER_KERNEL(KernelDefBuilder("Reshape")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 4)
                    .Provider(LotusIR::kCudaExecutionProvider)
                    .Alias(0, 0)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Reshape_1<float>);
}  // namespace Cuda
}  // namespace Lotus

#include "reshape.h"

namespace Lotus {
namespace Cuda {

#define REGISTER_KERNEL_TYPED(T)                                                        \
  REGISTER_KERNEL(KernelDefBuilder("Reshape")                                           \
                      .Domain(LotusIR::kOnnxDomain)                                     \
                      .SinceVersion(5)                                                  \
                      .Provider(LotusIR::kCudaExecutionProvider)                        \
                      .Alias(0, 0)                                                      \
                      .InputMemoryType<kMemTypeCPUInput>(1)                             \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())            \
                      .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()), \
                  Reshape<T>);                                                          \
                                                                                        \
  REGISTER_KERNEL(KernelDefBuilder("Reshape")                                           \
                      .Domain(LotusIR::kOnnxDomain)                                     \
                      .SinceVersion(1, 4)                                               \
                      .Provider(LotusIR::kCudaExecutionProvider)                        \
                      .Alias(0, 0)                                                      \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),           \
                  Reshape_1<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

}  // namespace Cuda
}  // namespace Lotus

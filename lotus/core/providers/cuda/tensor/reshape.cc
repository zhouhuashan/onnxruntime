#include "reshape.h"

namespace Lotus {
namespace Cuda {

#define REGISTER_KERNEL_TYPED(T)                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
    Reshape,                                                                  \
    kOnnxDomain,                                                              \
    5,                                                                        \
    T,                                                                        \
    kCudaExecutionProvider,                                                   \
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
                      .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()) \
                      .Alias(0, 0)                                            \
                      .InputMemoryType<kMemTypeCPUInput>(1),                  \
    Reshape<T>);                                                              \
                                                                              \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                          \
    Reshape_##T,                                                              \
    kOnnxDomain,                                                              \
    1,                                                                        \
    4,                                                                        \
    kCudaExecutionProvider,                                                   \
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
                      .Alias(0, 0),                                           \
    Reshape_1<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

}  // namespace Cuda
}  // namespace Lotus

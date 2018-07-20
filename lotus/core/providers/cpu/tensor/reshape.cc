#include "core/providers/cpu/tensor/reshape.h"
#include "core/inc/op_kernel_author.h"
namespace Lotus {

#define REGISTER_KERNEL_TYPED(T)                                                        \
ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                         \
    Reshape,                                                                            \
    5,                                                                                  \
    T,                                                                                  \
    KernelDefBuilder().Alias(0, 0)                                                      \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())            \
                      .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()), \
    Reshape<T>);                                                                        \
                                                                                        \
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(                                                     \
    Reshape_##T,                                                                        \
    1,                                                                                  \
    4,                                                                                  \
    KernelDefBuilder().Alias(0, 0)                                                      \
                      .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),           \
    Reshape_1<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

}  // namespace Lotus

#include "matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace Lotus {
namespace Cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  LOTUS_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), right_X->Shape()));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  LOTUS_RETURN_IF_NOT(Y->Location().name == CUDA, "Output should be allocated on CUDA");

  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        Base::CublasHandle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &one,
        reinterpret_cast<const CudaT*>(right_X->Data<T>()),
        static_cast<int>(helper.N()),
        reinterpret_cast<const CudaT*>(left_X->Data<T>()),
        static_cast<int>(helper.K()),
        &zero,
        reinterpret_cast<CudaT*>(Y->MutableData<T>()),
        static_cast<int>(helper.N())));
    return Status::OK();
  }

  CudaAsyncBuffer<const CudaT*> left_arrays(this, helper.LeftOffsets().size());
  CudaAsyncBuffer<const CudaT*> right_arrays(this, helper.RightOffsets().size());
  CudaAsyncBuffer<CudaT*> output_arrays(this, helper.OutputOffsets().size());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(left_X->Data<T>()), helper.LeftOffsets(), left_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(right_X->Data<T>()), helper.RightOffsets(), right_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<CudaT*>(Y->MutableData<T>()), helper.OutputOffsets(), output_arrays.CpuSpan());
  LOTUS_RETURN_IF_ERROR(left_arrays.CopyToGpu());
  LOTUS_RETURN_IF_ERROR(right_arrays.CopyToGpu());
  LOTUS_RETURN_IF_ERROR(output_arrays.CopyToGpu());

  // note that Lotus MLValue is row major, while cublas is column major,
  // so swap left/right operands
  CUBLAS_RETURN_IF_ERROR(cublasGemmBatchedHelper(
      Base::CublasHandle(),
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      static_cast<int>(helper.N()),
      static_cast<int>(helper.M()),
      static_cast<int>(helper.K()),
      &one,
      right_arrays.GpuPtr(),
      static_cast<int>(helper.N()),
      left_arrays.GpuPtr(),
      static_cast<int>(helper.K()),
      &zero,
      output_arrays.GpuPtr(),
      static_cast<int>(helper.N()),
      static_cast<int>(helper.OutputOffsets().size())));

  return Status::OK();
}

}  // namespace Cuda
}  // namespace Lotus

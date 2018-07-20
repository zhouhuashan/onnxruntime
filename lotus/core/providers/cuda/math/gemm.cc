#include "gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"

namespace Lotus {
namespace Cuda {

ONNX_OPERATOR_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

template <>
Status Gemm<float>::Compute(OpKernelContext* ctx) const {
  const auto X = ctx->Input<Tensor>(0);
  const auto W = ctx->Input<Tensor>(1);
  const auto B = ctx->Input<Tensor>(2);
  GemmHelper helper(X->Shape(), trans_A_, W->Shape(), trans_B_, B->Shape());

  if (!helper.State().IsOK())
    return helper.State();

  int M = gsl::narrow_cast<int>(helper.M());
  int N = gsl::narrow_cast<int>(helper.N());
  int K = gsl::narrow_cast<int>(helper.K());
  auto Y = ctx->Output(0, TensorShape(std::vector<int64_t>{M, N}));
  float* out_data = Y->template MutableData<float>();

  // broadcast bias if needed
  if (beta_ != 0) {
    auto& b_shape = B->Shape();
    const float* b_data = B->Data<float>();

    ;
    float one = 1;
    float zero = 0;
    if (b_shape.Size() == 1) {
      // if B is (), (1,) or (1, 1), broadcast the scalar
      CUBLAS_RETURN_IF_ERROR(cublasScopy(
          CublasHandle(),
          M * N,
          b_data,
          0,
          out_data,
          1));
    } else if (b_shape.NumDimensions() == 1 || b_shape[0] == 1) {
      // B is (N,) or (1, N), broadcast using Y(N,M) = 1 * B(N,1) x ones(1,M) + 0 * Y
      CUBLAS_RETURN_IF_ERROR(cublasSgemm(
          CublasHandle(),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          N, M, 1,
          /*alpha*/ &one,
          b_data, N,
          provider_->GetConstOnes(M), 1,
          /*beta*/ &zero,
          out_data, N));
    } else if (b_shape.NumDimensions() == 2 && b_shape[1] == 1) {
      // B is (M, 1), broadcast using Y(N,M) = 1 * ones(N,1) x B(1,M) + 0 * Y
      CUBLAS_RETURN_IF_ERROR(cublasSgemm(
          CublasHandle(),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          N, M, 1,
          /*alpha*/ &one,
          provider_->GetConstOnes(N), N,
          b_data, 1,
          /*beta*/ &zero,
          out_data, N));
    } else {
      // B is (M, N), no broadcast needed.
      CUDA_RETURN_IF_ERROR(cudaMemcpy(out_data, b_data, M * N * sizeof(float), cudaMemcpyDeviceToDevice));
    }
  }

  // Gemm, note that CUDA assumes col-major, so Y(N,M) = alpha * op(W) x op(X) + beta * Y
  CUBLAS_RETURN_IF_ERROR(cublasSgemm(
      CublasHandle(),
      trans_B_ ? CUBLAS_OP_T : CUBLAS_OP_N,
      trans_A_ ? CUBLAS_OP_T : CUBLAS_OP_N,
      N, M, K,
      &alpha_,
      W->Data<float>(), (trans_B_ ? K : N),
      X->Data<float>(), (trans_A_ ? M : K),
      &beta_,
      out_data, N));

  return Status::OK();
}

}  // namespace Cuda
}  // namespace Lotus

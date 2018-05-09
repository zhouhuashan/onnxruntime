#include "../cuda_common.h"
#include "matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace Lotus {
namespace Cuda {

REGISTER_KERNEL(KernelDefBuilder("MatMul")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCudaExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                MatMul<float>);

template <>
Status MatMul<float>::Compute(OpKernelContext* ctx) const {
  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper(left_X->Shape(), right_X->Shape());

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  const float alpha = 1.0f;
  const float beta = 0.0f;

  std::vector<const float*> left_arrays;
  std::vector<const float*> right_arrays;
  std::vector<float*> output_arrays;
  MatMulComputeHelper::OffsetToArrays(left_X->Data<float>(), helper.LeftOffsets(), left_arrays);
  MatMulComputeHelper::OffsetToArrays(right_X->Data<float>(), helper.RightOffsets(), right_arrays);
  MatMulComputeHelper::OffsetToArrays(Y->MutableData<float>(), helper.OutputOffsets(), output_arrays);

  // allocate temp memory for offset arrays
  AllocatorPtr alloc;
  LOTUS_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
  const float** left_arrays_cuda = (const float**)alloc->Alloc(left_arrays.size() * sizeof(const float*));
  const float** right_arrays_cuda = (const float**)alloc->Alloc(right_arrays.size() * sizeof(const float*));
  float** output_arrays_cuda = (float**)alloc->Alloc(output_arrays.size() * sizeof(float*));

  // note that Lotus MLValue is row major, while cublas is column major,
  // so swap left/right operands
  bool succeeded =
      CUDA_CALL(cudaMemcpy(left_arrays_cuda, left_arrays.data(), left_arrays.size() * sizeof(const float*), cudaMemcpyHostToDevice)) &&
      CUDA_CALL(cudaMemcpy(right_arrays_cuda, right_arrays.data(), right_arrays.size() * sizeof(const float*), cudaMemcpyHostToDevice)) &&
      CUDA_CALL(cudaMemcpy(output_arrays_cuda, output_arrays.data(), output_arrays.size() * sizeof(float*), cudaMemcpyHostToDevice)) &&
      CUBLAS_CALL(cublasSgemmBatched(
          Base::CublasHandle(),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          static_cast<int>(helper.N()),
          static_cast<int>(helper.M()),
          static_cast<int>(helper.K()),
          &alpha,
          right_arrays_cuda,
          static_cast<int>(helper.N()),
          left_arrays_cuda,
          static_cast<int>(helper.K()),
          &beta,
          output_arrays_cuda,
          static_cast<int>(helper.N()),
          static_cast<int>(helper.OutputOffsets().size())));

  alloc->Free(left_arrays_cuda);
  alloc->Free(right_arrays_cuda);
  alloc->Free(output_arrays_cuda);

  return succeeded ? Status::OK() : Status(SYSTEM, ENGINE_ERROR);
}

}  // namespace Cuda
}  // namespace Lotus

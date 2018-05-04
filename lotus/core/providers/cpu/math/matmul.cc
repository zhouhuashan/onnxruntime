#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/matmul.h"
#include "matmul_helper.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("MatMul")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                MatMul<float>);

template <>
Status MatMul<float>::Compute(OpKernelContext* ctx) const {
  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper(left_X->Shape(), right_X->Shape());

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // TODO: replace it with GemmBatch for performance, it's OK for now as GemmBatch unrolls as well
  for (int i = 0; i < helper.OutputOffsets().size(); i++) {
    Math::Gemm<float, CPUMathUtil>(
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(helper.M()),
        static_cast<int>(helper.N()),
        static_cast<int>(helper.K()),
        /* alpha */ 1.0f,
        left_X->Data<float>() + helper.LeftOffsets()[i],
        right_X->Data<float>() + helper.RightOffsets()[i],
        /* beta */ 0.0f,
        Y->MutableData<float>() + helper.OutputOffsets()[i],
        &CPUMathUtil::Instance());
  }

  return Status::OK();
}

}  // namespace Lotus

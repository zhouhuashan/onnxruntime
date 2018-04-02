#include "core/providers/cpu/misc/constant.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("Constant")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Constant<float>);

template <>
Status Constant<float>::Compute(OpKernelContext* ctx) const {
  std::vector<int64_t> dims{value_.dims().begin(), value_.dims().end()};

  TensorShape shape(dims);

  auto& C = *ctx->Output(0, shape);
  float* dest = C.MutableData<float>();
  for (float v : value_.float_data()) {
    *dest++ = v;
  }

  return Lotus::Common::Status::OK();
}

}  // namespace Lotus

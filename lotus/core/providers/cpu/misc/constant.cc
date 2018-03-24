#include "core/providers/cpu/misc/constant.h"

namespace Lotus {

template <>
Status Constant<float>::compute(OpKernelContext* ctx) const {
  std::vector<int64_t> dims;
  for (auto v : value_.dims())
    dims.push_back(v);
  TensorShape shape(dims);

  auto& C = *ctx->output(0, shape);
  float* pDest = C.mutable_data<float>();
  for (float v : value_.float_data()) {
    *pDest++ = v;
  }
  return Lotus::Common::Status::OK();
}

}  // namespace Lotus

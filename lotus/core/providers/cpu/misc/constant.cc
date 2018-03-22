#include "core/providers/cpu/misc/constant.h"

namespace Lotus {

template <>
void Constant<float>::compute(OpKernelContext* ctx) {
  std::vector<int64_t> dims;
  for (auto v : value_.dims())
    dims.push_back(v);
  TensorShape shape(dims);
  auto& C = *ctx->output(0, shape);
  float* pDest = C.mutable_data<float>();
  for (float v : value_.float_data())
    *pDest++ = v;
}

}  // namespace Lotus

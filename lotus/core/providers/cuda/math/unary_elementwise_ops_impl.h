#pragma once

namespace Lotus {
namespace Cuda {

// This macro simplifies coding to add a new op with following steps:
// 1. Add a new entry in UNARY_OPS() list
// 2. (optional) Define templated single element operator in unary_elementwise_ops_impl.cu
// 3. (optional) Implement specialized single element operator
// 4. Add op kernel class definition in unary_elementwise_ops.h
// 5. Add op kernel registration and compute specialization in unary_elementwise_ops.cc
// 6. Enable test of the newly added kernel on CUDA by changing test.Run() to test.RunOnCpuAndCuda()

#define UNARY_OPS()                          \
  UNARY_OP_NAME_EXPR(Abs, _Abs(a))           \
  UNARY_OP_NAME_EXPR(Neg, -a)                \
  UNARY_OP_NAME_EXPR(Ceil, _Ceil(a))         \
  UNARY_OP_NAME_EXPR(Floor, _Floor(a))       \
  UNARY_OP_NAME_EXPR(Reciprocal, InT(1) / a) \
  UNARY_OP_NAME_EXPR(Sqrt, _Sqrt(a))         \
  UNARY_OP_NAME_EXPR(Exp, _Exp(a))           \
  UNARY_OP_NAME_EXPR(Log, _Log(a))           \
  UNARY_OP_NAME_EXPR(Cast, (OutT)(a))

#define UNARY_ELEMENTWISE_IMPL_DECLARATION(name) \
  template <typename InT, typename OutT>         \
  void Impl_##name(                              \
      const InT* input_data,                     \
      OutT* output_data,                         \
      size_t count)

#define UNARY_OP_NAME_EXPR(name, expr) UNARY_ELEMENTWISE_IMPL_DECLARATION(name);
UNARY_OPS()
#undef UNARY_OP_NAME_EXPR

}  // namespace Cuda
}  // namespace Lotus

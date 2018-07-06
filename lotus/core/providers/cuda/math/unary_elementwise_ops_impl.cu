#include <cuda_runtime.h>
#include "unary_elementwise_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/elementwise_impl.cuh"

namespace Lotus {
namespace Cuda {

#define OP(name, expr)                                          \
  template <typename InT, typename OutT>                        \
  struct OP_##name {                                            \
    __device__ __inline__ OutT operator()(const InT& a) const { \
      return expr;                                              \
    }                                                           \
  };

#define UNARY_ELEMENTWISE_IMPL(name)             \
  UNARY_ELEMENTWISE_IMPL_DECLARATION(name) {     \
    UnaryElementWiseImpl(input_data,             \
                         output_data,            \
                         OP_##name<InT, OutT>(), \
                         count);                 \
  }

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL2(name, InT, OutT) \
  template void Impl_##name<InT, OutT>(const InT* input_data, OutT* output_data, size_t count);

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, T) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL2(name, T, T)

#define UNARY_OP_NAME_EXPR(name, expr) \
  OP(name, expr)                       \
  UNARY_ELEMENTWISE_IMPL(name)

UNARY_OPS()
#undef UNARY_OP_NAME_EXPR

// the postfix of means the types supported by the op:
// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// F: float
// D: double
// O: bool

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(name) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, half)     \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, float)    \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, double)

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL_CSILHFD(name) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int8_t)       \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int16_t)      \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int32_t)      \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, int64_t)      \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(name)

#define SPECIALIZED_UNARY_ELEMENTWISE_IMPL_BWUZCSILHFD(name) \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint8_t)          \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint16_t)         \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint32_t)         \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL(name, uint64_t)         \
  SPECIALIZED_UNARY_ELEMENTWISE_IMPL_CSILHFD(name)

SPECIALIZED_UNARY_ELEMENTWISE_IMPL_BWUZCSILHFD(Abs)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_CSILHFD(Neg)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Floor)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Ceil)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Reciprocal)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Sqrt)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Log)
SPECIALIZED_UNARY_ELEMENTWISE_IMPL_HFD(Exp)

SPECIALIZED_UNARY_ELEMENTWISE_IMPL2(Cast, uint32_t, int64_t)

}  // namespace Cuda
}  // namespace Lotus

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace Lotus {
namespace Cuda {

template <typename T, typename Tin>
void GatherImpl(
    const int64_t input_block_size,
    const int64_t indices_max,
    const Tin* indices_data,
    const fast_divmod* output_strides,
    const T* input_data,
    T* output_data,
    const size_t N);

}  // namespace Cuda
}  // namespace Lotus

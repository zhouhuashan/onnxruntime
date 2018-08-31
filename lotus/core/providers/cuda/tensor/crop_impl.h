#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace Lotus {
namespace Cuda {

template <typename T>
void CropImpl(
    const T* input_data,
    const int src_start_x,
    const int src_start_y,
    const int src_w,
    const int src_hw,
    const fast_divmod& fdm_dst_w,
    const fast_divmod& fdm_dst_hw,
    T* output_data,
    const size_t N);

}  // namespace Cuda
}  // namespace Lotus

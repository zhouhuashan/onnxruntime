#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace Lotus {
namespace Cuda {

template <typename T>
void ImageScalerImpl(
    const T* input_data,
    const float scale,
    const float* bias_data,
    const int64_t dims[4],
    T* output_data,
    const size_t N);

}  // namespace Cuda
}  // namespace Lotus

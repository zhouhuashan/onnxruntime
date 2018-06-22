#pragma once
#include <cuda_runtime.h>

namespace Lotus {
namespace Cuda {

void TileImpl(
    const size_t shape_rank,
    const int64_t* input_shape,
    const int64_t* input_stride,
    const float* input,
    const int64_t* output_stride,
    float* output,
    const size_t N);

}  // namespace Cuda
}  // namespace Lotus

#include "../cuda_kernel_common.cuh"
#include "tile_impl.h"

namespace Lotus {
namespace Cuda {

__global__ void _TileKernel(
    const size_t shape_rank,
    const int64_t* input_shape,
    const int64_t* input_stride,
    const float* input,
    const int64_t* output_stride,
    float* output,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;
  for (int dim = 0; dim < shape_rank; ++dim) {
    CUDA_LONG out_coord = output_index / output_stride[dim];
    output_index -= output_stride[dim] * out_coord;
    CUDA_LONG in_coord = out_coord % input_shape[dim];
    input_index += input_stride[dim] * in_coord;
  }
  output[id] = input[input_index];
}

void TileImpl(
    const size_t shape_rank,
    const int64_t* input_shape,
    const int64_t* input_stride,
    const float* input,
    const int64_t* output_stride,
    float* output,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _TileKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(shape_rank, input_shape, input_stride, input, output_stride, output, (CUDA_LONG)N);
}

}  // namespace Cuda
}  // namespace Lotus

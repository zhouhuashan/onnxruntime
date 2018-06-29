#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "common.cuh"

namespace Lotus {
namespace Cuda {

template <typename T, typename FuncT>
__global__ void _UnaryElementWise(
    const T* input_data,
    T* output_data,
    const FuncT& functor,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = functor(input_data[id]);
}

template <typename T, typename FuncT>
void UnaryElementWiseImpl(
    const T* input_data,
    T* output_data,
    const FuncT& func,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _UnaryElementWise<T, FuncT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      input_data,
      output_data,
      func,
      N);
}

// broadcast by computing output coordinate from offset, using fast_divmod
template <typename T, typename FuncT>
__global__ void _BinaryElementWise(
    size_t output_rank,
    bool lhs_dim0_broadcast,
    const int64_t* lhs_padded_strides,
    const T* lhs_data,
    bool rhs_dim0_broadcast,
    const int64_t* rhs_padded_strides,
    const T* rhs_data,
    const fast_divmod* fdm_output_strides,
    T* output_data,
    const FuncT& functor,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG lhs_index = 0, rhs_index = 0;
  // compute indexes with broadcasting rules: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
  CUDA_LONG offset = id;
  for (int dim = 0; dim < output_rank && offset != 0; dim++) {
    int q, r;
    fdm_output_strides[dim].divmod(offset, q, r);
    // compute index increase based on stride and broadcast
    // note that stride[i-1] == stride[i] means dim[i] is 1 (broadcasting)
    if ((dim == 0 && !lhs_dim0_broadcast) ||
        (dim > 0 && lhs_padded_strides[dim - 1] != lhs_padded_strides[dim]))
      lhs_index += static_cast<int>(lhs_padded_strides[dim]) * q;

    if ((dim == 0 && !rhs_dim0_broadcast) ||
        (dim > 0 && rhs_padded_strides[dim - 1] != rhs_padded_strides[dim]))
      rhs_index += static_cast<int>(rhs_padded_strides[dim]) * q;

    offset = r;
  }
  output_data[id] = functor(lhs_data[lhs_index], rhs_data[rhs_index]);
}

// for scalar broadcast or non-broadcast case
template <bool IncL, bool IncR, typename T, typename FuncT>
__global__ void _BinaryElementWiseSimple(
    const T* lhs_data,
    const T* rhs_data,
    T* output_data,
    FuncT func,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = func(lhs_data[IncL ? id : 0], rhs_data[IncR ? id : 0]);
}

template <typename T, typename FuncT>
void BinaryElementWiseNoBroadcastImpl(
    const T* lhs_data,
    const T* rhs_data,
    T* output_data,
    const FuncT& func,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _BinaryElementWiseSimple<true, true, T, FuncT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      lhs_data,
      rhs_data,
      output_data,
      func,
      N);
}

template <typename T, typename FuncT>
void BinaryElementWiseImpl(
    size_t output_rank_or_simple_broadcast,
    bool lhs_dim0_broadcast,
    const int64_t* lhs_padded_strides,
    const T* lhs_data,
    bool rhs_dim0_broadcast,
    const int64_t* rhs_padded_strides,
    const T* rhs_data,
    const fast_divmod* fdm_output_strides,
    T* output_data,
    const FuncT& func,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  if (output_rank_or_simple_broadcast == static_cast<size_t>(SimpleBroadcast::NoBroadcast)) {
    _BinaryElementWiseSimple<true, true, T, FuncT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        lhs_data,
        rhs_data,
        output_data,
        func,
        N);
  } else if (output_rank_or_simple_broadcast == static_cast<size_t>(SimpleBroadcast::LeftScalar)) {
    _BinaryElementWiseSimple<false, true, T, FuncT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        lhs_data,
        rhs_data,
        output_data,
        func,
        N);
  } else if (output_rank_or_simple_broadcast == static_cast<size_t>(SimpleBroadcast::RightScalar)) {
    _BinaryElementWiseSimple<true, false, T, FuncT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        lhs_data,
        rhs_data,
        output_data,
        func,
        N);
  } else {
    _BinaryElementWise<T, FuncT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        output_rank_or_simple_broadcast,
        lhs_dim0_broadcast,
        lhs_padded_strides,
        lhs_data,
        rhs_dim0_broadcast,
        rhs_padded_strides,
        rhs_data,
        fdm_output_strides,
        output_data,
        func,
        N);
  }
}

}  // namespace Cuda
}  // namespace Lotus

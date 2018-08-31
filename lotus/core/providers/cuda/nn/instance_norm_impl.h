#pragma once
#include "core/providers/cuda/shared_inc/fast_divmod.h"
namespace Lotus {
namespace Cuda {

template <typename T>
void InstanceNormImpl(
    const T* input_data,
    const T* scale,
    const T* bias,
    const T* mean,
    const T* variance,
    const double variance_correction,
    const double epsilon,
    const fast_divmod& fdm_HW,
    const fast_divmod& fdm_C,
    T* output_data,
    size_t count);

}  // namespace Cuda
}  // namespace Lotus

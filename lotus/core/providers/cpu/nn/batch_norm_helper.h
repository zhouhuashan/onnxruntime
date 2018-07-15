#pragma once

#include "core/common/status.h"
#include "core/framework/tensor.h"
#include <sstream>

namespace Lotus {
class BatchNormHelper {
 public:
  static Common::Status ValidateInputs(const Tensor* X,
                                       const Tensor* scale,
                                       const Tensor* B,
                                       const Tensor* mean,
                                       const Tensor* var) {
    // defined as per spec and used for validation
    constexpr int kNumInputScaleDimensions = 1;
    constexpr int kNumInputBiasDimensions = 1;
    constexpr int kNumInputMeanDimensions = 1;
    constexpr int kNumInputVarianceDimensions = 1;
    //constexpr int kMinCudaNumDims = 4;
    //constexpr int kMaxCudaNumDims = 5;

    if (X->Shape().GetDims().empty()) {
      return Common::Status(Common::LOTUS, Common::INVALID_ARGUMENT, "Invalid input X: Empty dimensions");
    }

    int64_t num_channels = X->Shape().GetDims()[1];

    if (scale->Shape().NumDimensions() != kNumInputScaleDimensions) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Invalid input scale: NumDimensions() != ", kNumInputScaleDimensions);
    }
    if (scale->Shape().GetDims()[0] != num_channels) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Invalid input scale: 0th dimension != ", num_channels);
    }

    if (B->Shape().NumDimensions() != kNumInputBiasDimensions) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Invalid input B: NumDimensions() != ", kNumInputBiasDimensions);
    }
    if (B->Shape().GetDims()[0] != num_channels) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Invalid input B: 0th dimension != ", num_channels);
    }

    if (mean->Shape().NumDimensions() != kNumInputMeanDimensions) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Invalid input mean: NumDimensions() != ", kNumInputMeanDimensions);
    }
    if (mean->Shape().GetDims()[0] != num_channels) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Invalid input mean: 0th dimension != ", num_channels);
    }

    if (var->Shape().NumDimensions() != kNumInputVarianceDimensions) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Invalid input var: NumDimensions() != ", kNumInputVarianceDimensions);
    }
    if (var->Shape().GetDims()[0] != num_channels) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Invalid input var: 0th dimension != ", num_channels);
    }

    return Common::Status::OK();
  }
};
}  // namespace Lotus

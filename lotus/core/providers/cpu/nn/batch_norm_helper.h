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
    if (X->Shape().GetDims().empty()) {
      std::ostringstream ostr;
      ostr << "Invalid input X: Empty dimensions";
      return Common::Status(LOTUS, INVALID_ARGUMENT, ostr.str());
    }

    int64_t num_channels = X->Shape().GetDims()[1];

    if (scale->Shape().NumDimensions() != kNumInputScaleDimensions) {
      std::ostringstream ostr;
      ostr << "Invalid input scale: NumDimensions() != " << kNumInputScaleDimensions;
      return Common::Status(LOTUS, INVALID_ARGUMENT, ostr.str());
    }
    if (scale->Shape().GetDims()[0] != num_channels) {
      std::ostringstream ostr;
      ostr << "Invalid input scale: 0th dimension != " << num_channels;
      return Common::Status(LOTUS, INVALID_ARGUMENT, ostr.str());
    }

    if (B->Shape().NumDimensions() != kNumInputBiasDimensions) {
      std::ostringstream ostr;
      ostr << "Invalid input B: NumDimensions() != " << kNumInputBiasDimensions;
      return Common::Status(LOTUS, INVALID_ARGUMENT, ostr.str());
    }
    if (B->Shape().GetDims()[0] != num_channels) {
      std::ostringstream ostr;
      ostr << "Invalid input B: 0th dimension != " << num_channels;
      return Common::Status(LOTUS, INVALID_ARGUMENT, ostr.str());
    }

    if (mean->Shape().NumDimensions() != kNumInputMeanDimensions) {
      std::ostringstream ostr;
      ostr << "Invalid input mean: NumDimensions() != " << kNumInputMeanDimensions;
      return Common::Status(LOTUS, INVALID_ARGUMENT, ostr.str());
    }
    if (mean->Shape().GetDims()[0] != num_channels) {
      std::ostringstream ostr;
      ostr << "Invalid input mean: 0th dimension != " << num_channels;
      return Common::Status(LOTUS, INVALID_ARGUMENT, ostr.str());
    }

    if (var->Shape().NumDimensions() != kNumInputVarianceDimensions) {
      std::ostringstream ostr;
      ostr << "Invalid input var: NumDimensions() != " << kNumInputVarianceDimensions;
      return Common::Status(LOTUS, INVALID_ARGUMENT, ostr.str());
    }
    if (var->Shape().GetDims()[0] != num_channels) {
      std::ostringstream ostr;
      ostr << "Invalid input var: 0th dimension != " << num_channels;
      return Common::Status(LOTUS, INVALID_ARGUMENT, ostr.str());
    }

    return Common::Status::OK();
  }

 private:
  // defined as per spec and used for validation
  static const int kNumInputScaleDimensions = 1;
  static const int kNumInputBiasDimensions = 1;
  static const int kNumInputMeanDimensions = 1;
  static const int kNumInputVarianceDimensions = 1;
  static const int kMinCudaNumDims = 4;
  static const int kMaxCudaNumDims = 5;
};
}  // namespace Lotus

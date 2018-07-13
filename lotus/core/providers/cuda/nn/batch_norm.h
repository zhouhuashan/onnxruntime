#pragma once

#include "gsl/gsl_util"
#include "core/providers/cuda/cuda_common.h"
#include <cfloat>  // FLT_EPSILON

namespace Lotus {
namespace Cuda {

template <typename T>
class BatchNorm final : public CudaKernel {
 public:
  BatchNorm(const OpKernelInfo& op_kernel_info)
      : CudaKernel{op_kernel_info},
        cudnn_batch_norm_mode_(CUDNN_BATCHNORM_SPATIAL) {
    float tmp_eplison;
    if (op_kernel_info.GetAttr<float>("epsilon", &tmp_eplison).IsOK()) {
      epsilon_ = tmp_eplison;
    }
    // Minimum allowed value is CUDNN_BN_MIN_EPSILON defined in cudnn.h.
    if (epsilon_ <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOGS_DEFAULT(WARNING) << "Provided epsilon is smaller than CUDNN_BN_MIN_EPSILON. Setting it to CUDNN_BN_MIN_EPSILON";
    }
    epsilon_ = std::max(epsilon_, CUDNN_BN_MIN_EPSILON);

    // spatial or not
    int64_t tmp_spatial;
    if (op_kernel_info.GetAttr<int64_t>("spatial", &tmp_spatial).IsOK()) {
      spatial_ = tmp_spatial;
    }

    if (spatial_ == 0) {
      cudnn_batch_norm_mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;  // TODO add test case for this when implemented in CPU as well.
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  double epsilon_;
  int64_t spatial_ = 1;  // default as per spec
  cudnnBatchNormMode_t cudnn_batch_norm_mode_;
};

}  // namespace Cuda
}  // namespace Lotus

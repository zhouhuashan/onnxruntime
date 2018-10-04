// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/upsample.h"

namespace onnxruntime {
namespace cuda {

struct TVMState;

template <typename T>
class Upsample : public UpsampleBase, public CudaKernel {
 public:
  Upsample(OpKernelInfo info) : UpsampleBase(info), CudaKernel(info) {
    // only support scale HW for NCHW input
    if (scales_.size() > 2) {
      for (int i = 0; i < scales_.size() - 2; i++) {
        ONNXRUNTIME_ENFORCE(scales_[i] == 1, "Can only upsample in H/W channels.");
      }
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<TVMState> s_;
  mutable std::mutex mutex;
};

}  // namespace cuda
}  // namespace onnxruntime

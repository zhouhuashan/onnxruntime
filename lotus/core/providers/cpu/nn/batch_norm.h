/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#pragma once

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/autopad_type.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

template <typename T>
class BatchNorm final : public OpKernel {
 public:
  BatchNorm(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    LOTUS_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());

    // keeping the below for reference. we don't need these for inference.
    //LOTUS_ENFORCE(op_kernel_info.GetAttr<int64_t>("is_test", &is_test_).IsOK());
    //LOTUS_ENFORCE(op_kernel_info.GetAttr<float>("momentum", &momentum_).IsOK());
    //LOTUS_ENFORCE(op_kernel_info.GetAttr<int64_t>("spatial", &spatial_).IsOK());
  }

  Status ValidateInputs(const Tensor* X,
                        const Tensor* scale,
                        const Tensor* B,
                        const Tensor* mean,
                        const Tensor* var) const;

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  float epsilon_;
  int64_t is_test_;  // ignored in this implementation since we're doing inferencing only.
  float momentum_;   // ignored in this implementation since we're doing inferencing only.
  int64_t spatial_;  // ignored in this implementation since we're doing inferencing only.

  // defined as per spec and used for validation
  const int kNumInputXDimensions = 4;
  const int kNumInputScaleDimensions = 1;
  const int kNumInputBiasDimensions = 1;
  const int kNumInputMeanDimensions = 1;
  const int kNumInputVarianceDimensions = 1;
};
}  // namespace Lotus

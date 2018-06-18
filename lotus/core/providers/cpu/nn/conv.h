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

#include "core/providers/cpu/nn/conv_base.h"

namespace Lotus {
namespace {
// helper function
inline void ComputeSizeAndPad(
    const int64_t in_dim,
    const int64_t stride,
    const int64_t kernel,
    const int64_t dilation,
    AutoPadType pad_type,
    int64_t* pad_head,
    int64_t* pad_tail,
    int64_t* out_dim) {
  const int64_t dkernel = dilation * (kernel - 1) + 1;

  if (pad_type == AutoPadType::NOTSET) {
    *out_dim = static_cast<int64_t>(static_cast<float>(in_dim + *pad_head + *pad_tail - dkernel) / stride + 1);
  } else {
    switch (pad_type) {
      case AutoPadType::VALID:
        *pad_head = 0;
        *pad_tail = 0;
        *out_dim = (in_dim - dkernel) / stride + 1;
        break;
      case AutoPadType::SAME_UPPER:
      case AutoPadType::SAME_LOWER: {
        LOTUS_ENFORCE(dilation == 1, "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.");
        int64_t legacy_target_size = (in_dim + stride - 1) / stride;
        int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
        *out_dim = (in_dim + pad_needed - dkernel) / stride + 1;

        if (pad_type == AutoPadType::SAME_LOWER) {
          *pad_head = (pad_needed + 1) / 2;
        } else {
          *pad_head = pad_needed / 2;
        }
        *pad_tail = pad_needed - *pad_head;
      } break;
      default:
        throw NotImplementedException("pad type not supported");
    }
  }
}
}  // namespace

template <typename T>
class Conv : public ConvBase {
 public:
  Conv(const OpKernelInfo& info) : ConvBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  void InferOutputShape(const TensorShape& input_shape,
                        const vector<int64_t>& kernel_shape,
                        const vector<int64_t>& strides,
                        const vector<int64_t>& dilations,
                        vector<int64_t>* pads,
                        vector<int64_t>* output_shape) const {
    for (int dim = 0; dim < input_shape.NumDimensions(); ++dim) {
      int64_t dim_size = 0;
      ComputeSizeAndPad(input_shape[dim],
                        strides[dim],
                        kernel_shape[dim],
                        dilations[dim],
                        auto_pad_,
                        &pads->at(dim),
                        &pads->at(input_shape.NumDimensions() + dim),
                        &dim_size);
      output_shape->push_back(dim_size);
    }
  }
};

}  // namespace Lotus

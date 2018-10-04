// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T>
struct Slice final : OpKernel {
  Slice(const OpKernelInfo& info) : OpKernel(info) {
    has_axes_ = info.GetAttrs("axes", axes_).IsOK();

    if (!info.GetAttrs("starts", starts_).IsOK())
      ONNXRUNTIME_THROW("Invalid 'starts' attribute value");
    if (!info.GetAttrs("ends", ends_).IsOK())
      ONNXRUNTIME_THROW("Invalid 'ends' attribute value");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::vector<int64_t> axes_;
  bool has_axes_;
  std::vector<int64_t> starts_, ends_;
};  // namespace onnxruntime

}  // namespace onnxruntime

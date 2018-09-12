#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/autopad_type.h"

namespace onnxruntime {

template <typename T>
class RoiPool : public OpKernel {
 public:
  RoiPool(const OpKernelInfo& info) : OpKernel(info) {
    std::vector<int64_t> pooled_shape;
    LOTUS_ENFORCE(info.GetAttrs<int64_t>("pooled_shape", pooled_shape).IsOK());
    LOTUS_ENFORCE(pooled_shape.size() == 2);

    pooled_height_ = pooled_shape[0];
    pooled_width_ = pooled_shape[1];
    LOTUS_ENFORCE(pooled_height_ > 0);
    LOTUS_ENFORCE(pooled_width_ > 0);

    LOTUS_ENFORCE(info.GetAttr<float>("spatial_scale", &spatial_scale_).IsOK());
    LOTUS_ENFORCE(spatial_scale_ > 0);
  }

  ~RoiPool() override = default;

  Status Compute(OpKernelContext* context) const override;

 protected:
  int64_t pooled_height_, pooled_width_;
  float spatial_scale_;
};
}  // namespace onnxruntime

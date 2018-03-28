#ifndef CORE_PROVIDERS_CPU_MATH_CLIP_H
#define CORE_PROVIDERS_CPU_MATH_CLIP_H

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

template <typename T>
class Clip final : public OpKernel {
 public:
  Clip(const OpKernelInfo& info) : OpKernel(info) {
    if (!op_kernel_info_.GetAttr<T>("max", &max_).IsOK()) {
      has_max_ = false;
    }
    if (!op_kernel_info_.GetAttr<T>("min", &min_).IsOK()) {
      has_min_ = false;
    }
  }

  Status compute(OpKernelContext* context) const override;

 private:
  T max_, min_;
  bool has_max_ = true;
  bool has_min_ = true;
};

}  // namespace Lotus

#endif  // !CORE_PROVIDERS_CPU_MATH_CLIP_H

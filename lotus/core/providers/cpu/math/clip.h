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
      max_ = std::numeric_limits<T>::max();
    }
    if (!op_kernel_info_.GetAttr<T>("min", &min_).IsOK()) {
      min_ = std::numeric_limits<T>::min();
    }
  }

  void compute(OpKernelContext* context) override;

 private:
  T max_, min_;
};

}  // namespace Lotus

#endif  // !CORE_PROVIDERS_CPU_MATH_CLIP_H

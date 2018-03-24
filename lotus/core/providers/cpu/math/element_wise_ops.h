#ifndef CORE_PROVIDERS_CPU_MATH_ELEMENT_WISE_OPS_H
#define CORE_PROVIDERS_CPU_MATH_ELEMENT_WISE_OPS_H

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

template <typename T>
class Add final : public OpKernel {
 public:
  Add(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public OpKernel {
 public:
  Sub(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public OpKernel {
 public:
  Mul(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Reciprocal final : public OpKernel {
 public:
  Reciprocal(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

template <typename T>
class Sum final : public OpKernel {
 public:
  Sum(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status compute(OpKernelContext* context) const override;
};

}  // namespace Lotus

#endif  // !CORE_PROVIDERS_CPU_MATH_ELEMENT_WISE_OPS_H

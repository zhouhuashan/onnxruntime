#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(Elu,
                                       {
                                         EIGEN_X_VAR(xm);
                                         EIGEN_Y = (xm >= 0).select(xm, Attr("alpha") * (xm.exp() - 1));
                                       },
                                       {"alpha"})

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(LeakyRelu,
                                       {
                                         EIGEN_X_VAR(xm);
                                         EIGEN_Y = (xm >= 0).select(xm, Attr("alpha") * xm);
                                       },
                                       {"alpha"})

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(Relu,
                                       { EIGEN_Y = EIGEN_X.cwiseMax(0); },
                                       {})

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(Sigmoid,
                                       {
                                         EIGEN_X_VAR(xm);
                                         EIGEN_Y_VAR(ym);
                                         ym = (xm >= 0).select(1 / (1. + (-xm.abs()).exp()), 1 - 1 / (1. + (-xm.abs()).exp()));
                                       },
                                       {})

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(HardSigmoid,
                                       {
                                         EIGEN_X_VAR(xm);
                                         EIGEN_Y_VAR(ym);
                                         ym = ((Attr("alpha") * xm + Attr("beta")).cwiseMin(1.0f)).cwiseMax(0.0f);
                                       },
                                       (std::vector<std::string>{"alpha", "beta"}))

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(Tanh, { EIGEN_Y = EIGEN_X.tanh(); }, {})

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu,
                                       {
                                         EIGEN_X_VAR(xm);
                                         EIGEN_Y = (xm >= Attr("alpha")).select(xm, 0);
                                       },
                                       {"alpha"})

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(Selu,
                                       {
                                         EIGEN_X_VAR(xm);
                                         EIGEN_Y = Attr("gamma") * (xm.cwiseMax(0.0f) + (Attr("alpha") * (xm.array().exp() - 1.0f)).cwiseMin(0.0f));
                                       },
                                       {"alpha", "gamma"})

template <typename T>
class PRelu final : public OpKernel {
 public:
  PRelu(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace Lotus
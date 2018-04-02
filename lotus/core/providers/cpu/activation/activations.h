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

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(Tanh, { EIGEN_Y = EIGEN_X.tanh(); }, {})

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu,
                                       {
                                         EIGEN_X_VAR(xm);
                                         EIGEN_Y = (xm >= Attr("alpha")).select(xm, 0);
                                       },
                                       {"alpha"})

}  // namespace Lotus

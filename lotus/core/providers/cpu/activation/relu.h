#ifndef CORE_PROVIDERS_CPU_ACTIVATION_RELU_H
#define CORE_PROVIDERS_CPU_ACTIVATION_RELU_H

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(Relu, { EIGEN_Y = EIGEN_X.cwiseMax(0); })
}

#endif  // !CORE_PROVIDERS_CPU_ACTIVATION_RELU_H
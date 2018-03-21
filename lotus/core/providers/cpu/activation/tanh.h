#ifndef CORE_PROVIDERS_CPU_ACTIVATION_TANH_H
#define CORE_PROVIDERS_CPU_ACTIVATION_TANH_H

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(Tanh, { EIGEN_Y = EIGEN_X.tanh(); })
}

#endif  // !CORE_PROVIDERS_CPU_ACTIVATION_TANH_H

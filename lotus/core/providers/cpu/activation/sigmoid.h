#ifndef CORE_PROVIDERS_CPU_ACTIVATION_SIGMOID_H
#define CORE_PROVIDERS_CPU_ACTIVATION_SIGMOID_H

#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

    DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(Sigmoid,
        {
            EIGEN_X_VAR(xM);
            EIGEN_Y_VAR(yM);
            yM = 1 / ( 1. + (-xM.abs()).exp());
            yM = (1 + xM.cwiseSign()) / 2 * yM + (1 - xM.cwiseSign()) / 2 * ( 1 - yM);
        })
}

#endif // !CORE_PROVIDERS_CPU_ACTIVATION_SIGMOID_H

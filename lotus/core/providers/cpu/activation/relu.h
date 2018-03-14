#ifndef CORE_PROVIDERS_CPU_ACTIVATION_RELU_H
#define CORE_PROVIDERS_CPU_ACTIVATION_RELU_H

#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

    template<typename T>
    class ReLU final : public OpKernel {

    public:
        ReLU(const OpKernelInfo& info, const KernelDef* kernel_def) : OpKernel(info, kernel_def)
        {
        }

        void compute(OpKernelContext* context) override;
    };

}

#endif // !CORE_PROVIDERS_CPU_ACTIVATION_RELU_H
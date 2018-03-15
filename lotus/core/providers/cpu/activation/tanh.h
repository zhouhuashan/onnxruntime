#ifndef CORE_PROVIDERS_CPU_ACTIVATION_TANH_H
#define CORE_PROVIDERS_CPU_ACTIVATION_TANH_H

#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

    template<typename T>
    class Tanh final : public OpKernel {

    public:
        Tanh(const OpKernelInfo& info) : OpKernel(info)
        {
        }

        void compute(OpKernelContext* context) override;
    };

}

#endif // !CORE_PROVIDERS_CPU_ACTIVATION_TANH_H

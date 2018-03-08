#ifndef CORE_KERNELS_CPU_MATH_CLIP_H
#define CORE_KERNELS_CPU_MATH_CLIP_H

#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

template<typename T>
class Clip final : public OpKernel {

public:
    Clip(OpKernelInfo* info): OpKernel(info)
    {
        if (!info->GetAttr<T>("max", &max_).IsOK()) {
             max_ = std::numeric_limits<T>::max();
        }
        if (!info->GetAttr<T>("min", &min_).IsOK()){
            min_ = std::numeric_limits<T>::min();
        }
    }

    void compute(OpKernelContext* context) override;

private:
    T max_, min_;
};

}

#endif // !CORE_KERNELS_CPU_MATH_CLIP_H

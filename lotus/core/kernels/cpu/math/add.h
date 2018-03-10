#pragma once
#ifndef CORE_KERNELS_CPU_MATH_ADD_H
#define CORE_KERNELS_CPU_MATH_ADD_H

#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

template<typename T>
struct Constant final : OpKernel {

    Constant(OpKernelInfo* info): OpKernel(info)
    {
#if 0
        if (!info->GetAttr("value", &value_).IsOK()) {
             LOTUS_ENFORCE(false, "Must have valid 'value' attribute");
        }
#endif
    }

    void compute(OpKernelContext* context) override;

private:
    Tensor* value_;
};

template<typename T>
struct Concat final : OpKernel {

    Concat(OpKernelInfo* info) : OpKernel(info)
    {
        if (!info->GetAttr("axis", &axis_).IsOK()) {
            LOTUS_ENFORCE(false, "Must have valid 'axis' attribute");
        }
    }

    void compute(OpKernelContext* context) override;

private:
    int axis_;
};


template<typename T>
struct Add final : OpKernel {

    Add(OpKernelInfo* info) : OpKernel(info)
    {
    }

    void compute(OpKernelContext* context) override;

private:
};

template<typename T>
struct Sub final : OpKernel {

    Sub(OpKernelInfo* info) : OpKernel(info)
    {
    }

    void compute(OpKernelContext* context) override;

private:
};

template<typename T>
struct Mul final : OpKernel {

    Mul(OpKernelInfo* info) : OpKernel(info)
    {
    }

    void compute(OpKernelContext* context) override;

private:
};

template<typename T>
struct Sum final : OpKernel {

    Sum(OpKernelInfo* info) : OpKernel(info)
    {
    }

    void compute(OpKernelContext* context) override;

private:
};

}

#endif // !CORE_KERNELS_CPU_MATH_CLIP_H

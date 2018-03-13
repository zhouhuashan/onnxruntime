#pragma once
#ifndef CORE_PROVIDERS_CPU_MATH_BASIC_H
#define CORE_PROVIDERS_CPU_MATH_BASIC_H

#include "core/framework/op_kernel.h"
#include "core/common/common.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

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
struct Reciprocal final : OpKernel {

    Reciprocal(OpKernelInfo* info) : OpKernel(info)
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

#endif // !CORE_PROVIDERS_CPU_MATH_BASIC_H

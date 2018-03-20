/**
* Derived from caffe2, need copy right annoucement here.
*/

/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef LOTUS_UTILS_MATH_CPU_H_
#define LOTUS_UTILS_MATH_CPU_H_

#include "Eigen/Core"
#include "Eigen/Dense"

namespace Lotus {

    // Common Eigen types that we will often use
    template <typename T>
    using EigenMatrixMap =
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;
    template <typename T>
    using EigenArrayMap =
        Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> >;
    template <typename T>
    using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >;
    template <typename T>
    using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> >;
    template <typename T>
    using ConstEigenMatrixMap =
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >;
    template <typename T>
    using ConstEigenArrayMap =
        Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> >;
    template <typename T>
    using ConstEigenVectorMap =
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1> >;
    template <typename T>
    using ConstEigenVectorArrayMap =
        Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1> >;

    class CPUMathUtil
    {
    public:
        /*CPUMathUtil contains some help method like generate a
        random seed. We only need a single instance for it.*/
        static CPUMathUtil& Instance() {
            static CPUMathUtil p;
            return p;
        }
        //todo: the random generate interface.
    private:
        CPUMathUtil() {}
    };

#define EIGEN_Y EigenVectorArrayMap<T>(Y->mutable_data<T>(), Y->shape().Size())
#define EIGEN_X ConstEigenVectorArrayMap<T>(X->data<T>(), X->shape().Size())
#define EIGEN_X_VAR(var) ConstEigenVectorArrayMap<T> var(X->data<T>(), X->shape().Size())
#define EIGEN_Y_VAR(var) EigenVectorArrayMap<T> var(Y->mutable_data<T>(), Y->shape().Size())

#define DECLARE_EIGEN_UNARY_ELEMENTWISE_KERNEL(class_name, func)    \
    template<typename T>                                            \
    class class_name final : public OpKernel {                      \
    public:                                                         \
        static const char* TypeTraits() {                           \
            return #class_name;                                     \
        }                                                           \
                                                                    \
        class_name(const OpKernelInfo& info) : OpKernel(info)       \
        {}                                                          \
                                                                    \
        void compute(OpKernelContext* context) override {           \
            const Tensor* X = context->template input<Tensor>(0);   \
            Tensor* Y = context->output(0, X->shape());             \
            func;                                                   \
        }                                                           \
    };

}
#endif
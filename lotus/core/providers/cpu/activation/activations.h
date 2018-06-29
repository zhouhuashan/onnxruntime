#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {

#define EIGEN_X ConstEigenVectorArrayMap<T>(X->Data<T>(), X->Shape().Size())
#define EIGEN_X_VAR(var) ConstEigenVectorArrayMap<T> var(X->Data<T>(), X->Shape().Size())
#define EIGEN_Y EigenVectorArrayMap<T>(Y->MutableData<T>(), Y->Shape().Size())
#define EIGEN_Y_VAR(var) EigenVectorArrayMap<T> var(Y->MutableData<T>(), Y->Shape().Size())

template <typename T>
class Elu final : public OpKernel {
 public:
  Elu(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttr("alpha", &alpha_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (xm >= 0).select(xm, (T)alpha_ * (xm.exp() - 1));
    return Status::OK();
  }

 private:
  float alpha_ = 1.0f;
};

template <typename T>
class HardSigmoid final : public OpKernel {
 public:
  HardSigmoid(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttr("alpha", &alpha_);
    info.GetAttr("beta", &beta_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y_VAR(ym);
    ym = (((T)alpha_ * xm + (T)beta_).cwiseMin(1.0f)).cwiseMax(0.0f);
    return Status::OK();
  }

 private:
  float alpha_ = 0.2f;
  float beta_ = 0.5f;
};

template <typename T>
class LeakyRelu final : public OpKernel {
 public:
  LeakyRelu(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttr("alpha", &alpha_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (xm >= 0).select(xm, (T)alpha_ * xm);
    return Status::OK();
  }

 private:
  float alpha_ = 0.01f;
};

template <typename T>
class ParametricSoftplus final : public OpKernel {
 public:
  ParametricSoftplus(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttr("alpha", &alpha_);
    info.GetAttr("beta", &beta_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (T)alpha_ * ((xm * (T)beta_).exp() + 1.0f).log();
    return Status::OK();
  }

 private:
  float alpha_ = 1.0f;
  float beta_ = 1.0f;
};

template <typename T>
class Relu final : public OpKernel {
 public:
  Relu(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_Y = EIGEN_X.cwiseMax(0);
    return Status::OK();
  }
};

template <typename T>
class ScaledTanh final : public OpKernel {
 public:
  ScaledTanh(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttr("alpha", &alpha_);
    info.GetAttr("beta", &beta_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_Y = (T)alpha_ * (EIGEN_X * (T)beta_).tanh();
    return Status::OK();
  }

 private:
  float alpha_ = 1.0f;
  float beta_ = 1.0f;
};

template <typename T>
class Selu final : public OpKernel {
 public:
  Selu(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttr("alpha", &alpha_);
    info.GetAttr("gamma", &gamma_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (T)gamma_ * (xm.cwiseMax(0.0f) + ((T)alpha_ * (xm.array().exp() - 1.0f)).cwiseMin(0.0f));
    return Status::OK();
  }

 private:
  float alpha_ = 1.67326319217681884765625f;
  float gamma_ = 1.05070102214813232421875f;
};

template <typename T>
class Sigmoid final : public OpKernel {
 public:
  Sigmoid(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y_VAR(ym);
    ym = (xm >= 0).select(1 / (1. + (-xm.abs()).exp()), 1 - 1 / (1. + (-xm.abs()).exp()));
    return Status::OK();
  }
};

template <typename T>
class Softsign final : public OpKernel {
 public:
  Softsign(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (1 + xm.abs()).inverse() * xm;
    return Status::OK();
  }
};

template <typename T>
class Tanh final : public OpKernel {
 public:
  Tanh(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_Y = EIGEN_X.tanh();
    return Status::OK();
  }
};

template <typename T>
class ThresholdedRelu final : public OpKernel {
 public:
  ThresholdedRelu(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttr("alpha", &alpha_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (xm > (T)alpha_).select(xm, 0);
    return Status::OK();
  }

 private:
  float alpha_ = 1.0f;
};

}  // namespace Lotus

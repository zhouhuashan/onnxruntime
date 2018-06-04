#pragma once

#include <random>
#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

class RandomNormal final : public OpKernel {
 public:
  RandomNormal(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<float>("mean", &mean_).IsOK());
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<float>("scale", &scale_).IsOK());

    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (!op_kernel_info_.GetAttr<float>("seed", &seed).IsOK()) {
      seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    int64_t dtype;
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<int64_t>("dtype", &dtype).IsOK());
    dtype_ = static_cast<TensorProto::DataType>(dtype);
    LOTUS_ENFORCE(TensorProto::DataType_IsValid(dtype_) && dtype_ != TensorProto::UNDEFINED,
                  "Invalid dtype of ", dtype_);

    std::vector<int64_t> shape;
    LOTUS_ENFORCE(op_kernel_info_.GetAttrs<int64_t>("shape", shape).IsOK());
    shape_ = TensorShape(shape);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float mean_;
  float scale_;
  std::default_random_engine generator_;
  TensorProto::DataType dtype_;
  TensorShape shape_;
};

class RandomNormalLike final : public OpKernel {
 public:
  RandomNormalLike(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<float>("mean", &mean_).IsOK());
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<float>("scale", &scale_).IsOK());

    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (!op_kernel_info_.GetAttr<float>("seed", &seed).IsOK()) {
      seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    int64_t dtype;
    if (op_kernel_info_.GetAttr<int64_t>("dtype", &dtype).IsOK()) {
      dtype_ = static_cast<TensorProto::DataType>(dtype);
      LOTUS_ENFORCE(TensorProto::DataType_IsValid(dtype_) && dtype_ != TensorProto::UNDEFINED,
                    "Invalid dtype of ", dtype_);
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float mean_;
  float scale_;
  std::default_random_engine generator_;
  TensorProto::DataType dtype_ = TensorProto::DataType::TensorProto_DataType_UNDEFINED;  //optional and may be inferred
};

class RandomUniform final : public OpKernel {
 public:
  RandomUniform(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<float>("high", &high_).IsOK());
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<float>("low", &low_).IsOK());

    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (!op_kernel_info_.GetAttr<float>("seed", &seed).IsOK()) {
      seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    int64_t dtype;
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<int64_t>("dtype", &dtype).IsOK());
    dtype_ = static_cast<TensorProto::DataType>(dtype);
    LOTUS_ENFORCE(TensorProto::DataType_IsValid(dtype_) && dtype_ != TensorProto::UNDEFINED,
                  "Invalid dtype of ", dtype_);

    std::vector<int64_t> shape;
    LOTUS_ENFORCE(op_kernel_info_.GetAttrs<int64_t>("shape", shape).IsOK());
    shape_ = TensorShape(shape);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float high_;
  float low_;
  std::default_random_engine generator_;
  TensorProto::DataType dtype_;
  TensorShape shape_;
};

class RandomUniformLike final : public OpKernel {
 public:
  RandomUniformLike(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<float>("high", &high_).IsOK());
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<float>("low", &low_).IsOK());
    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (!op_kernel_info_.GetAttr<float>("seed", &seed).IsOK()) {
      seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    int64_t dtype;
    if (op_kernel_info_.GetAttr<int64_t>("dtype", &dtype).IsOK()) {
      dtype_ = static_cast<TensorProto::DataType>(dtype);
      LOTUS_ENFORCE(TensorProto::DataType_IsValid(dtype_) && dtype_ != TensorProto::UNDEFINED,
                    "Invalid dtype of ", dtype_);
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float high_;
  float low_;
  std::default_random_engine generator_;
  TensorProto::DataType dtype_ = TensorProto::DataType::TensorProto_DataType_UNDEFINED;  //optional and may be inferred
};

class Multinomial final : public OpKernel {
 public:
  Multinomial(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(op_kernel_info_.GetAttr<int64_t>("sample_size", &num_samples_).IsOK());

    float seed = 0.f;
    if (!op_kernel_info_.GetAttr<float>("seed", &seed).IsOK()) {
      seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};

    int64_t output_dtype_tmp;
    if (!op_kernel_info_.GetAttr<int64_t>("dtype", &output_dtype_tmp).IsOK()) {
      output_dtype_ = TensorProto_DataType_INT32;  // default is INT32 as per spec
    } else {
      output_dtype_ = static_cast<TensorProto::DataType>(output_dtype_tmp);
    }
    LOTUS_ENFORCE(TensorProto::DataType_IsValid(output_dtype_) && output_dtype_ != TensorProto::UNDEFINED,
                  "Invalid dtype of ", output_dtype_);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  int64_t num_samples_;
  std::default_random_engine generator_;
  TensorProto::DataType output_dtype_;
};
}  // namespace Lotus

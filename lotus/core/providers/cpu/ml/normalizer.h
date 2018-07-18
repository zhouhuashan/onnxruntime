#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/ml/ml_common.h"

#include "gsl/span"

namespace Lotus {
namespace ML {

class Normalizer final : public OpKernel {
 public:
  Normalizer(const OpKernelInfo& info) : OpKernel(info) {
    std::string norm;
    LOTUS_ENFORCE(info.GetAttr<std::string>("norm", &norm).IsOK());

    normalization_ = MakeNormalize(norm);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  void Normalize(OpKernelContext* context) const;

  NORMALIZE normalization_;
};

}  // namespace ML
}  // namespace Lotus

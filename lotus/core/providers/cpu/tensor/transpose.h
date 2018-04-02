#pragma once

#include "gsl/gsl_util"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
template <typename T>
class Transpose final : public OpKernel {
 public:
  Transpose(const OpKernelInfo& info) : OpKernel{info}, perm_specified_(false) {
    Status status = info.GetAttrs<int64_t>("perm", perm_);

    if (status.IsOK()) {
      perm_specified_ = true;
      size_t rank = perm_.size();
      std::vector<bool> seen(rank, false);
      // Check that perm_ is a valid permutation of [0,rank-1]
      for (auto i : perm_) {
        if ((i < 0) || (i >= gsl::narrow<int64_t>(rank)))
          LOTUS_THROW("Attribute perm of Transpose has an invalid value. Value ", i, " is outside range.");
        if (seen[i])
          LOTUS_THROW("Attribute perm of Transpose has an invalid value. Value ", i, " is repeated.");
        seen[i] = true;
      }
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool perm_specified_;
  std::vector<int64_t> perm_;
};
}  // namespace Lotus

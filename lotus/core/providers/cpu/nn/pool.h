#pragma once

#include "core/providers/cpu/nn/pool_base.h"

namespace Lotus {

class AveragePool;

class MaxPool;

struct PoolProcessContext {
  int64_t p_;
  PoolProcessContext() : p_(2) {}
};

template <typename T, typename PoolType>
class Pool final : public PoolBase {
 public:
  Pool(OpKernelInfo info) : PoolBase(info) {
    const std::string& op_name = info.GetKernelDef().OpName();
    if (op_name == "LpPool" || op_name == "GlobalLpPool") {
      info.GetAttr<int64_t>("p", &pool_context_.p_);
    }
  }

  ~Pool() = default;

  Status Compute(OpKernelContext* context) const override;

 private:
  PoolProcessContext pool_context_;
};

}  // namespace Lotus

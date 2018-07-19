#pragma once

#include "core/providers/cpu/nn/pool_base.h"

namespace Lotus {

class AveragePool;

class MaxPool;

struct PoolProcessContext {
  int64_t p_{2};
  PoolProcessContext() {}
};

template <typename T, typename PoolType>
class Pool final : public OpKernel, public PoolBase {
 public:
  Pool(OpKernelInfo info) : OpKernel(info), PoolBase(info) {
    const std::string& op_name = info.GetKernelDef().OpName();
    if (op_name == "LpPool" || op_name == "GlobalLpPool") {
      info.GetAttr<int64_t>("p", &pool_context_.p_);
    }
  }

  ~Pool() override{};

  Status Compute(OpKernelContext* context) const override;

 private:
  PoolProcessContext pool_context_;
};

}  // namespace Lotus

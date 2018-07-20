#pragma once

#include "core/providers/cpu/nn/pool_base.h"

namespace Lotus {

class LpPool;

class PoolProcessContext {
 private:
  int64_t p_;

 public:
  friend class LpPool;
  PoolProcessContext() {}
  void init(const OpKernelInfo& info) {
    LOTUS_ENFORCE(info.GetAttr<int64_t>("p", &p_).IsOK());
  }
};

template <typename T, typename PoolType>
class Pool final : public OpKernel, public PoolBase {
 public:
  Pool(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {
    const std::string& op_name = info.GetKernelDef().OpName();
    if (op_name == "LpPool" || op_name == "GlobalLpPool") {
      pool_context_.init(info);
    }
  }

  ~Pool() override{};

  Status Compute(OpKernelContext* context) const override;

 private:
  PoolProcessContext pool_context_;
};

}  // namespace Lotus

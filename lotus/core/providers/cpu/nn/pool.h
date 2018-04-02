#pragma once

#include "core/providers/cpu/nn/pool_base.h"

namespace Lotus {

class AveragePool;

class MaxPool;

template <typename T, typename PoolType>
class Pool final : public PoolBase {
 public:
  Pool(OpKernelInfo info) : PoolBase(info) {}

  ~Pool() = default;

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace Lotus

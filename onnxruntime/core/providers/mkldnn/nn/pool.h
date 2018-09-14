// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace mkl_dnn {

enum PoolType {
  MaxPool,
  AveragePool
};

template <typename T, PoolType type>
class Pool final : public OpKernel, public PoolBase {
 public:
  Pool(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace mkl_dnn
}  // namespace onnxruntime

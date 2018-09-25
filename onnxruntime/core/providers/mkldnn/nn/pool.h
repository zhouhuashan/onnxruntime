// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T, typename PoolType>
class Pool final : public onnxruntime::Pool<T, PoolType> {
 public:
  Pool(const OpKernelInfo& info) : onnxruntime::Pool<T, PoolType>(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace mkl_dnn
}  // namespace onnxruntime

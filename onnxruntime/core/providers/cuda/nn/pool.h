// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace cuda {

enum PoolType {
  MaxPool,
  AveragePool,
  MaxPool_8 // MaxPool of opset 8
};

template <typename T, PoolType type>
class Pool final : public CudaKernel, public PoolBase {
 public:
  Pool(OpKernelInfo info) : CudaKernel(info), PoolBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};
}  // namespace cuda
}  // namespace onnxruntime

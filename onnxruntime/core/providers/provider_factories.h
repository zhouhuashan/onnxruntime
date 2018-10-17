// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/execution_provider.h"

namespace onnxruntime {

// Information needed to construct CPU/MKL-DNN execution providers.
struct CPUExecutionProviderInfo {
  bool create_arena{true};

  explicit CPUExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}
  CPUExecutionProviderInfo() = default;
};

// Create CPU execution provider
std::unique_ptr<IExecutionProvider>
CreateBasicCPUExecutionProvider(const CPUExecutionProviderInfo& info);

// Create MKL-DNN execution provider
std::unique_ptr<IExecutionProvider>
CreateMKLDNNExecutionProvider(const CPUExecutionProviderInfo& info);

// Information needed to construct CUDA execution providers.
struct CUDAExecutionProviderInfo {
  int device_id{0};
};

// Create cuda execution provider
std::unique_ptr<IExecutionProvider>
CreateCUDAExecutionProvider(const CUDAExecutionProviderInfo& info);

}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/provider_factories.h"

#include "core/providers/mkldnn/mkldnn_execution_provider.h"

namespace onnxruntime {

// Create MKL-DNN execution provider
std::unique_ptr<IExecutionProvider> CreateMKLDNNExecutionProvider(const CPUExecutionProviderInfo& info) {
  return std::make_unique<MKLDNNExecutionProvider>(info);
}

}  // namespace onnxruntime

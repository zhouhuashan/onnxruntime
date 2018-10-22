// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/provider_factories.h"
#include "core/providers/brainslice/brain_slice_execution_provider.h"

namespace onnxruntime {

// Create nuphar execution provider
std::unique_ptr<IExecutionProvider>
CreateBrainSliceExecutionProvider(const fpga::FPGAInfo& info) {
  return std::make_unique<brainslice::BrainSliceExecutionProvider>(info);
}

}  // namespace onnxruntime

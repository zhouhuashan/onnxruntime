// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/provider_factories.h"
#include "core/providers/nuphar/nuphar_execution_provider.h"

namespace onnxruntime {

// Create nuphar execution provider
std::unique_ptr<IExecutionProvider>
CreateNupharExecutionProvider(const NupharExecutionProviderInfo& info) {
  return std::make_unique<NupharExecutionProvider>(info);
}

}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/execution_provider.h"

namespace onnxruntime {

// Information needed to construct Nuphar execution providers.
struct NupharExecutionProviderInfo {
  std::string name;
  int device_id{0};
  // By default, construct "stackvm" TVM target, for which the default device_type is kDLCPU.
  std::string target_str{"stackvm"};

  explicit NupharExecutionProviderInfo(const std::string& provider_name,
                                       int dev_id = 0,
                                       const std::string& tgt_str = "stackvm")
      : name(provider_name), device_id(dev_id), target_str(tgt_str) {}
  NupharExecutionProviderInfo() = default;
};

// Create Nuphar execution provider
std::unique_ptr<IExecutionProvider>
CreateNupharExecutionProvider(const NupharExecutionProviderInfo& info);

}  // namespace onnxruntime

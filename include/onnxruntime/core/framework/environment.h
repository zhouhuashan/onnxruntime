// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/common/common.h"
#include "core/common/status.h"

namespace onnxruntime {
/**
Provides the runtime environment for onnxruntime.    
Create one instance for the duration of execution.
*/
class Environment {
 public:
  /**
  Create and initialize the runtime environment.
  */
  static Status Create(std::unique_ptr<Environment>& environment) {
    environment = std::unique_ptr<Environment>(new Environment());
    auto status = environment->Initialize();
    return status;
  }

  ~Environment();

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(Environment);

  Environment() = default;
  Status Initialize();
};
}  // namespace onnxruntime

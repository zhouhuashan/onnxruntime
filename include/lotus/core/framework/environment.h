#pragma once

#include <memory>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/allocatormgr.h"

namespace Lotus {
/**
Provides the runtime environment for Lotus.    
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
}  // namespace Lotus

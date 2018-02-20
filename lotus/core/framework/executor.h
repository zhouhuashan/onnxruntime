#ifndef CORE_FRAMEWORK_EXECUTOR_H
#define CORE_FRAMEWORK_EXECUTOR_H

#include <vector>
#include "core/framework/execution_provider.h"
#include "core/framework/ml_value.h"
#include "core/graph/status.h"

namespace Lotus
{
  class Executor
  {
  public:
    Executor();
    ~Executor();

    
  private:
    const Graph* graph_;    
    ExecutionFrame* root_frame;
  }
}

#endif  // CORE_FRAMEWORK_EXECUTOR_H

#ifndef CORE_FRAMEWORK_EXECUTOR_H
#define CORE_FRAMEWORK_EXECUTOR_H

#include <vector>
#include "core/framework/ml_value.h"
#include "core/framework/execution_frame.h"
#include "core/graph/graph.h"
#include "core/common/status.h"

namespace Lotus
{
  class Executor
  {
  public:
    Executor() {}
    ~Executor() {}

    
  private:
    const LotusIR::Graph* graph_;
    ExecutionFrame* root_frame_;
  };
}

#endif  // CORE_FRAMEWORK_EXECUTOR_H

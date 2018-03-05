#ifndef LOTUS_CORE_FRAMEWORK_EXECUTOR_H_
#define LOTUS_CORE_FRAMEWORK_EXECUTOR_H_

#include <vector>
#include "core/framework/ml_value.h"
#include "core/framework/execution_frame.h"
#include "core/graph/graph.h"
#include "core/common/status.h"
#include "core/framework/session_state.h"
#include "core/framework/inference_session.h"

namespace Lotus
{
  class Executor
  {
  public:
    virtual ~Executor() {}

    static std::unique_ptr<Executor> NewSequentialExecutor(const SessionState& session_state);
    static std::unique_ptr<Executor> NewParallelExecutor(const SessionState& session_state);    

    virtual Common::Status Execute(const RunOptions& run_options,
                                   const std::vector<MLValue>& feeds,
                                   std::vector<MLValue>* p_fetches);
    
 protected:
    Executor() {}
    
    // TODO: Should we use naked pointer here?
    // If yes, explain the ownership and lifetime
    const LotusIR::Graph* graph_;
    // TODO: Should we use naked pointer here?
    // If yes, explain the ownership and lifetime
    ExecutionFrame* root_frame_;
  };
}

#endif  // LOTUS_CORE_FRAMEWORK_EXECUTOR_H_

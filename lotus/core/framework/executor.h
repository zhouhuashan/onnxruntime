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

    static std::unique_ptr<Executor> NewSequentialExecutor(const SessionState& session_state,
                                                           std::unique_ptr<ExecutionFrame> p_exec_frame);

    virtual Common::Status Execute(const RunOptions& run_options,
                                   const NameMLValMap& feeds,
                                   const std::vector<std::string>& output_names,
                                   std::vector<MLValue>* p_fetches);
    
 protected:
    Executor(std::unique_ptr<ExecutionFrame> p_exec_frame)
        : root_frame_(std::move(p_exec_frame)) {
    }
    std::unique_ptr<ExecutionFrame> root_frame_;
  };
}

#endif  // LOTUS_CORE_FRAMEWORK_EXECUTOR_H_

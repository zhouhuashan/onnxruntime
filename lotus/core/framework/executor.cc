#include "core/framework/executor.h"

#include <thread>
#include <chrono>

namespace Lotus
{
Common::Status Executor::Execute(const RunOptions& run_options,
                                 const std::vector<MLValue>& feeds,
                                 std::vector<MLValue>* p_fetches) {
  UNUSED_PARAMETER(run_options);
  UNUSED_PARAMETER(feeds);
  UNUSED_PARAMETER(p_fetches);
  return Common::Status::OK();
}

// TODO move to its own file
class SequentialExecutor: public Executor {
 public:
  SequentialExecutor(const SessionState& session_state) {
    UNUSED_PARAMETER(session_state);
    // 1. partition the graph
    // 2. sort the graph in topological order
    // 3. for each node in the resulting graph, call the respective execution provider's
    //    Compute() method.
    //    this assumes the execution provider info is present in the node as an attribute
  }
  
  Common::Status Execute(const RunOptions& run_options,
                         const std::vector<MLValue>& feeds,
                         std::vector<MLValue>* p_fetches) override {
    // TODO
    UNUSED_PARAMETER(run_options);
    UNUSED_PARAMETER(feeds);
    UNUSED_PARAMETER(p_fetches);
    
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1s); // TODO remove this artificial wait; it is here only for testing
    
    return Common::Status::OK();
  }
};

// TODO move to its own file
class ParallelExecutor: public Executor {
 public:
  ParallelExecutor(const SessionState& session_state) {
    UNUSED_PARAMETER(session_state);
  }
  
  Common::Status Execute(const RunOptions& run_options,
                         const std::vector<MLValue>& feeds,
                         std::vector<MLValue>* p_fetches) override {
    // TODO
    UNUSED_PARAMETER(run_options);
    UNUSED_PARAMETER(feeds);
    UNUSED_PARAMETER(p_fetches);    
    return Common::Status::OK();
  }  
};

std::unique_ptr<Executor> Executor::NewSequentialExecutor(const SessionState& session_state) {
  std::unique_ptr<Executor> retval;
  retval.reset(new SequentialExecutor(session_state));
  return retval;
}

std::unique_ptr<Executor> Executor::NewParallelExecutor(const SessionState& session_state) {
  std::unique_ptr<Executor> retval;
  retval.reset(new ParallelExecutor(session_state));
  return retval;
}
}

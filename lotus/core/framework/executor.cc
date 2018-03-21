#include "core/framework/executor.h"

#include <chrono>
#include <thread>

#include "core/framework/session_state.h"

namespace Lotus {
Common::Status Executor::Execute(const RunOptions& run_options,
                                 const NameMLValMap& feeds,
                                 const std::vector<std::string>& output_names,
                                 std::vector<MLValue>* p_fetches) {
  UNUSED_PARAMETER(run_options);
  UNUSED_PARAMETER(feeds);
  UNUSED_PARAMETER(output_names);
  UNUSED_PARAMETER(p_fetches);
  return Common::Status::OK();
}

// TODO move to its own file
class SequentialExecutor : public Executor {
 public:
  SequentialExecutor(const SessionState& session_state,
                     const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names)
      : root_frame_(feeds, output_names, session_state),
        session_state_(session_state) {
  }

  Common::Status Execute(const RunOptions& run_options,
                         const NameMLValMap& feeds,
                         const std::vector<std::string>& output_names,
                         std::vector<MLValue>* p_fetches) override {
    UNUSED_PARAMETER(run_options);
    UNUSED_PARAMETER(output_names);
    UNUSED_PARAMETER(p_fetches);
    UNUSED_PARAMETER(feeds);

    // TODO initialize the execution frame with feeds, output_names, p_fetches and model weights
    // TODO write test for executor when execution frame is ready

    // we added the kernels in the topological order when we initialized
    // the session
    for (auto& p_op_kernel : session_state_.GetKernelVector()) {
      if (!p_op_kernel) {
        // TODO continue for now since we don't have any kernels ready
        // when the kernels are ready, we should log and error here and
        // return with fail status.
        continue;
      }
      // get execution provider for this node
      const Node& node = p_op_kernel->node();
      const std::string& exec_provider_name = node.GetExecutionProvider();
      IExecutionProvider* p_exec_provider = session_state_.GetExecutionProvider(exec_provider_name);

      // construct OpKernelContext
      OpKernelContext op_kernel_context(&root_frame_, p_op_kernel.get());

      // call Compute on the execution provider
      LOTUS_RETURN_IF_ERROR(p_exec_provider->Compute(node, &op_kernel_context));
    }

    return Common::Status::OK();
  }

 private:
  ExecutionFrame root_frame_;
  const SessionState& session_state_;
};

std::unique_ptr<Executor> Executor::NewSequentialExecutor(const SessionState& session_state,
                                                          const NameMLValMap& feeds,
                                                          const std::vector<std::string>& output_names) {
  return std::unique_ptr<Executor>(new SequentialExecutor(session_state, feeds, output_names));
}
}  // namespace Lotus

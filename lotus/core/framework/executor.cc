#include "core/framework/executor.h"

#include <chrono>
#include <thread>

#include "core/framework/allocation_planner.h"
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
                       std::unique_ptr<ExecutionFrame> p_exec_frame)
        : Executor(std::move(p_exec_frame)),
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

      // TODO write test for executor when execution frame is ready

      // we added the kernels in the topological order when we initialized
      // the session
      const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();

      for (auto& node_exec_plan : p_seq_exec_plan->execution_plan) {
        auto node_index = node_exec_plan.node_index;
        OpKernel* p_op_kernel = session_state_.GetKernel(node_index);
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
        OpKernelContext op_kernel_context(root_frame_.get(), p_op_kernel);

        // call Compute on the execution provider
        LOTUS_RETURN_IF_ERROR(p_exec_provider->Compute(node, &op_kernel_context));

        // free ml-values corresponding to this node
        for (auto i = node_exec_plan.free_from_index; i <= node_exec_plan.free_to_index; ++i) {
          auto mlvalue_idx = p_seq_exec_plan->to_be_freed[i];
          // TODO perform delete here
          root_frame_->ReleaseMLValue(mlvalue_idx);
        }
      }

      return Common::Status::OK();
    }

   private:
    const SessionState& session_state_;
  };

  std::unique_ptr<Executor> Executor::NewSequentialExecutor(const SessionState& session_state,
                                                            std::unique_ptr<ExecutionFrame> p_exec_frame) {
    return std::unique_ptr<Executor>(new SequentialExecutor(session_state, std::move(p_exec_frame)));
  }
}  // namespace Lotus

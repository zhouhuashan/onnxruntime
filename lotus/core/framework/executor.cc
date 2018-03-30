#include "core/framework/executor.h"

#include <chrono>
#include <thread>

#include "core/common/logging/logging.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/session_state.h"

namespace Lotus {
Common::Status Executor::Execute(const RunOptions& run_options,
                                 const Logging::Logger& run_logger,
                                 const NameMLValMap& feeds,
                                 const std::vector<std::string>& output_names,
                                 std::vector<MLValue>* p_fetches) {
  UNUSED_PARAMETER(run_options);
  UNUSED_PARAMETER(run_logger);
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
                         const Logging::Logger& run_logger,
                         const NameMLValMap& feeds,
                         const std::vector<std::string>& output_names,
                         std::vector<MLValue>* p_fetches) override {
    UNUSED_PARAMETER(run_options);
    UNUSED_PARAMETER(feeds);

    const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
    const auto& exec_plan_vec = p_seq_exec_plan->execution_plan;

    for (int i = 0; i < exec_plan_vec.size(); ++i) {
      const auto& node_exec_plan = exec_plan_vec[i];
      auto node_index = node_exec_plan.node_index;
      auto p_op_kernel = session_state_.GetKernel(node_index);

      LOTUS_ENFORCE(p_op_kernel != nullptr);  // if a kernel has been added in the session state, it better be NON-null.

      // construct OpKernelContext
      OpKernelContext op_kernel_context(&root_frame_, p_op_kernel, run_logger);

      // call compute on the kernel
      // TODO Today the kernels don't return any status code.
      // They throw exceptions instead. We should change the compute
      // method to return a status code.
      p_op_kernel->compute(&op_kernel_context);

      // free ml-values corresponding to this node
      ReleaseNodeMLValues(p_seq_exec_plan, node_exec_plan);
    }

    LOTUS_RETURN_IF_ERROR(FetchOutput(output_names, p_fetches));

    return Common::Status::OK();
  }

  Common::Status FetchOutput(const std::vector<std::string>& output_names,
                             std::vector<MLValue>* p_fetches) {
    for (const auto& oname : output_names) {
      int mlvalue_index;
      LOTUS_RETURN_IF_ERROR(session_state_.GetMLValueIdx(oname, &mlvalue_index));
      const MLValue& output_mlvalue = root_frame_.GetMLValue(mlvalue_index);
      p_fetches->push_back(output_mlvalue);
    }

    return Common::Status::OK();
  }

  void ReleaseNodeMLValues(const SequentialExecutionPlan* p_seq_exec_plan,
                           const SequentialExecutionPlan::NodeExecutionPlan& node_exec_plan) {
    for (auto i = node_exec_plan.free_from_index; i <= node_exec_plan.free_to_index; ++i) {
      auto mlvalue_idx = p_seq_exec_plan->to_be_freed[i];
      root_frame_.ReleaseMLValue(mlvalue_idx);
    }
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

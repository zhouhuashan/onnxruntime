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
    UNUSED_PARAMETER(feeds);
    UNUSED_PARAMETER(output_names);
    UNUSED_PARAMETER(p_fetches);

    // TODO write test for executor when execution frame is ready

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

      const Node& node = p_op_kernel->node();

      // allocate the inputs and outputs of this node as per the plan
      AllocateNodeArgs(node, node.InputDefs(), p_seq_exec_plan->allocation_plan);
      AllocateNodeArgs(node, node.OutputDefs(), p_seq_exec_plan->allocation_plan);

      // get execution provider for this node
      IExecutionProvider* p_exec_provider = session_state_.GetExecutionProvider(node.GetExecutionProvider());

      // construct OpKernelContext
      OpKernelContext op_kernel_context(&root_frame_, p_op_kernel);

      // call Compute on the execution provider
      LOTUS_RETURN_IF_ERROR(p_exec_provider->Compute(node, &op_kernel_context));

      // free ml-values corresponding to this node
      ReleaseNodeMLValues(p_seq_exec_plan, node_exec_plan);
    }

    return Common::Status::OK();
  }

  void ReleaseNodeMLValues(const SequentialExecutionPlan* p_seq_exec_plan,
                           const SequentialExecutionPlan::NodeExecutionPlan& node_exec_plan) {
    for (auto i = node_exec_plan.free_from_index; i <= node_exec_plan.free_to_index; ++i) {
      auto mlvalue_idx = p_seq_exec_plan->to_be_freed[i];
      // TODO perform delete here
      root_frame_.ReleaseMLValue(mlvalue_idx);
    }
  }

  // this function doesn't belong to the class
  static void GetDimensionsFromTensorShapeProto(const onnx::TensorShapeProto* p_tensor_shape_proto,
                                                std::vector<int64_t>* dims) {
    for (int index = 0; index < p_tensor_shape_proto->dim_size(); index++)
      dims->push_back(p_tensor_shape_proto->dim(index).dim_value());
  }

  Common::Status AllocateNodeArgs(const Node& node,
                                  const std::vector<NodeArg*>& node_args,
                                  const std::vector<SequentialExecutionPlan::AllocPlanPerValue>& alloc_plan) {
    for (auto& elem : node_args) {
      const std::string& name = elem->Name();

      // get mlvalue index using name from the session_state
      int mlvalue_index;
      LOTUS_RETURN_IF_ERROR(session_state_.GetMLValueIdx(name, &mlvalue_index));

      // get AllocPlanPerValue from alloc_plan using this mlvalue_index
      if (mlvalue_index < 0 || mlvalue_index >= alloc_plan.size()) {
        std::ostringstream ostr;
        ostr << "Argument with name: " << name << " with mlvalue_index: " << mlvalue_index << " does not exist in the alloc_plan";
        return Common::Status(Common::LOTUS, Common::FAIL, ostr.str());
      }
      const auto& per_alloc_plan = alloc_plan[mlvalue_index];

      // get allocator info
      IExecutionProvider* p_exec_provider = session_state_.GetExecutionProvider(node.GetExecutionProvider());
      const AllocatorInfo& alloc_info = p_exec_provider->GetTempSpaceAllocator().Info();

      // use the AllocPlanPerValue to perform allocation
      AllocKind alloc_kind = per_alloc_plan.alloc_kind;
      std::vector<int64_t> dims;
      GetDimensionsFromTensorShapeProto(elem->Shape(), &dims);
      TensorShape shape(dims);

      MLDataType ml_data_type = DataTypeImpl::GetType<float>();  // assume float?

      switch (alloc_kind) {
        case AllocKind::kAllocate: {
          LOTUS_RETURN_IF_ERROR(root_frame_.AllocateMLValueTensorSelfOwnBuffer(mlvalue_index,
                                                                               ml_data_type,
                                                                               alloc_info,
                                                                               shape));
          break;
        }
        case AllocKind::kReuse: {
          int reuse_mlvalue_index = per_alloc_plan.reused_buffer;
          LOTUS_RETURN_IF_ERROR(root_frame_.AllocateMLValueTensorPreAllocateBuffer(mlvalue_index,
                                                                                   reuse_mlvalue_index,
                                                                                   ml_data_type,
                                                                                   alloc_info,
                                                                                   shape));
          break;
        }
        default:
          return Common::Status(Common::LOTUS, Common::FAIL, "Invalid allocation kind");
      };
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

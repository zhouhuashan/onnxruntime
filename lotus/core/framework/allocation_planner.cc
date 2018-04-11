#include "allocation_planner.h"
#include <list>
#include <unordered_map>
#include "core/common/exceptions.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/session_state.h"
#include "core/graph/utils.h"
#include "core/framework/data_types.h"

/*
TODO: 
- (Not for milestone 1)
- handle different types of devices:
  - identify placement of all tensors and ml-values
  - insert copies between different devices as required
*/

namespace Lotus {

class PlannerImpl {
 private:
  const SessionState* p_session_state_;
  const ISequentialPlannerContext* p_context_;
  SequentialExecutionPlan* plan_;

  // MLValueInfo: Auxiliary information about an MLValue used only during plan-generation:
  struct MLValueInfo {
    const LotusIR::NodeArg* p_def_site;  // the unique NodeArg where the MLValue is assigned a value
    int usecount = 0;                    // static reference-count
    MLValueIndex reused_buffer_index;    // index of original buffer to reuse
  };

  // ml_value_info_ is indexed by an MLValueIndex
  std::vector<MLValueInfo> ml_value_info_;

  // FreeBufferInfo is used to track information about ml-values whose buffers are
  // free to be reused.
  struct FreeBufferInfo {
    MLValueIndex ml_value;
    // deallocate_point is an index into the execution-plan; thus, ml_value becomes free after
    // this step in the execution-plan is completed.
    int deallocate_point;
    FreeBufferInfo(MLValueIndex mlvalue, int dealloc_point) : ml_value(mlvalue), deallocate_point(dealloc_point) {}
  };
  // freelist_ : a list of ml-values whose buffers are free to be reused, sorted by when
  // they became free (more recently freed earlier in the list).
  std::list<FreeBufferInfo> freelist_;

  MLValueIndex index(const MLValueName& name) {
    MLValueIndex result;
    auto status = p_session_state_->GetMLValueIdx(name, &result);
    LOTUS_ENFORCE(status.IsOK());
    return result;
  }

  int& UseCount(MLValueIndex n) { return ml_value_info_.at(n).usecount; }
  int& UseCount(const MLValueName& name) { return UseCount(index(name)); }

  MLValueIndex& Buffer(MLValueIndex n) { return ml_value_info_.at(n).reused_buffer_index; }

  SequentialExecutionPlan::AllocPlanPerValue& AllocPlan(MLValueIndex n) { return plan_->allocation_plan.at(n); }

  // Initialize state for a given ml-value at its definition site:
  void ProcessDef(const LotusIR::NodeArg* p_def_site) {
    const MLValueName& name = p_def_site->Name();
    MLValueIndex id = index(name);
    MLValueInfo& info = ml_value_info_.at(id);
    info.usecount = 0;
    info.reused_buffer_index = index(name);  // initially, no reuse; the ml-value uses its own buffer
    info.p_def_site = p_def_site;
  }

  void Reuse(MLValueIndex reused, MLValueIndex reused_for) {
    // find original buffer underlying ml-value we want to reuse:
    MLValueIndex original = Buffer(reused);
    // record that the new buffer will reuse that original buffer
    Buffer(reused_for) = original;
    // adjust original buffer's usecount
    UseCount(original) += UseCount(reused_for);

    // update allocation plan (for use at execution-time)
    auto& symplan = AllocPlan(reused_for);
    symplan.alloc_kind = AllocKind::kReuse;
    symplan.reused_buffer = original;
  }

  // Find if there exists some input tensor that we can use in-place for output_arg
  bool FindReusableInput(const LotusIR::Node& node, int output_arg_num, MLValueIndex* reusable_input) {
    auto p_output_arg = node.OutputDefs()[output_arg_num];
    auto p_opkernel = p_session_state_->GetKernel(node.Index());
    if (nullptr == p_opkernel) return false;
    const KernelDef& kernel_def = p_opkernel->KernelDef();
    const std::vector<std::pair<int, int>>& alias_map = kernel_def.Alias();
    for (auto pair : alias_map) {
      if (pair.second == output_arg_num) {
        // we _must_ reuse this input to satisfy aliasing requirement: (e.g., for reshape)
        auto p_input_arg = node.InputDefs()[pair.first];
        *reusable_input = index(p_input_arg->Name());
        return true;
      }
    }

    const std::vector<std::pair<int, int>>& inplace_map = kernel_def.MayInplace();
    for (auto pair : inplace_map) {
      if (pair.second == output_arg_num) {
        auto p_input_arg = node.InputDefs()[pair.first];
        auto input_arg_index = index(p_input_arg->Name());
        auto original = Buffer(input_arg_index);
        if (1 == UseCount(original)) {
          if (SameSize(*p_input_arg, *p_output_arg)) {
            // we can reuse this input since it is its last use and permitted for in-place update
            *reusable_input = input_arg_index;  // or original; both should be okay
            return true;
          }
        }
      }
    }
    return false;
  }

  bool SameShape(const TensorShapeProto& shape1, const TensorShapeProto& shape2) {
    // TODO: shape-inference does not exist yet; it may be better to have shape inference
    // return our own TensorShape instead of TensorShapeProto ... and we can use equality
    // defined in TensorShape.
    int rank1 = shape1.dim_size();
    if (shape2.dim_size() != rank1) return false;
    for (int i = 0; i < rank1; i++) {
      auto val1 = shape1.dim(i);
      auto val2 = shape2.dim(i);
      if (val1.has_dim_value() && val2.has_dim_value() && (val1.dim_value() == val2.dim_value()))
        continue;  // same known dimension
      if (val1.has_dim_param() && val2.has_dim_param() && (val1.dim_param() == val2.dim_param()))
        continue;  // same unknown dimension
      return false;
    }
    return true;
  }

  MLDataType GetMLDataType(const LotusIR::NodeArg& arg) {
    const DataType ptype = arg.Type();
    const onnx::TypeProto& type_proto = onnx::Utils::DataTypeUtils::ToTypeProto(ptype);
    return DataTypeImpl::TypeFromProto(type_proto);
  }

  /*! \brief Given a tensor-type, return the size of an element of the tensor.
  */
  size_t GetElementSize(const DataType& tensor_type) {
    const onnx::TypeProto& type_proto = onnx::Utils::DataTypeUtils::ToTypeProto(tensor_type);
    MLDataType ml_data_type = DataTypeImpl::TypeFromProto(type_proto);
    const TensorTypeBase* tensor_type_base = ml_data_type->AsTensorType();
    LOTUS_ENFORCE(nullptr != tensor_type_base);
    MLDataType elt_type = tensor_type_base->GetElementType();
    return elt_type->Size();
  }

  bool SameSize(const TensorShapeProto& shape1, const DataType& ptype1,
                const TensorShapeProto& shape2, const DataType& ptype2) {
    return (GetElementSize(ptype1) == GetElementSize(ptype2)) && SameShape(shape1, shape2);

    /* TODO: we can generalize this if the concrete shapes are known for both:
    if (KnownSize(p_shape1) && KnownSize(p_shape2)) {
      // Comparison of statically-known size
      auto size1 = NumElements(p_shape1) * EltSize(ptype1);
      auto size2 = NumElements(p_shape2) * EltSize(ptype2);
      return size1 == size2;
    } else {
      // Comparison of statically-unknown size buffers
      return SameElementSize(ptype1, ptype2) && SameShape(shape1, shape2);
    }
    */
  }

  bool SameSize(const LotusIR::NodeArg& arg1, const LotusIR::NodeArg& arg2) {
    if ((!arg1.Exists()) || (!arg2.Exists())) return false;
    auto p_shape1 = p_context_->GetShape(arg1);
    auto p_shape2 = p_context_->GetShape(arg2);
    // If the shapes are unknown, we conservatively assume they may be of different size.
    if ((nullptr == p_shape1) || (nullptr == p_shape2)) return false;
    return SameSize(*p_shape1, arg1.Type(), *p_shape2, arg2.Type());
  }

  // Find if freelist contains a buffer of the same size as output_arg
  bool FindReusableTensor(const LotusIR::NodeArg& output_arg, MLValueIndex* reusable_tensor) {
    auto p_required_buffer_shape = p_context_->GetShape(output_arg);
    if (nullptr == p_required_buffer_shape) return false;
    auto required_buffer_type = output_arg.Type();

    for (auto it = freelist_.begin(); it != freelist_.end(); ++it) {
      auto reusable = it->ml_value;
      auto p_node_arg = ml_value_info_.at(reusable).p_def_site;
      auto p_available_buffer_shape = p_context_->GetShape(*p_node_arg);
      if (nullptr != p_available_buffer_shape) {
        auto available_buffer_type = p_node_arg->Type();
        if (SameSize(*p_available_buffer_shape, available_buffer_type, *p_required_buffer_shape, required_buffer_type)) {
          *reusable_tensor = reusable;
          freelist_.erase(it);
          return true;
        }
      }
    }
    return false;
  }

  void Initialize(size_t num_graph_nodes, size_t num_ml_values) {
    // All ml-value indices must be in range 0 .. num_ml_values-1
    ml_value_info_.resize(num_ml_values);

    // Initialize execution plan:
    plan_->execution_plan.clear();
    plan_->execution_plan.reserve(num_graph_nodes);

    // Initialize allocation plan:
    plan_->allocation_plan.clear();
    plan_->allocation_plan.resize(num_ml_values);
  }

  void ComputeUseCounts(const LotusIR::Graph& graph,
                        std::vector<SequentialExecutionPlan::NodeExecutionPlan>& execution_plan) {
    // Note: for every ml-value, its definition must appear before all its uses in a topological sort of a valid model

    for (auto graph_input : graph.GetInputs()) {
      ProcessDef(graph_input);
      UseCount(graph_input->Name())++;  // Models caller's usage post-inference; ensures it will not be reused.
    }

    for (SequentialExecutionPlan::NodeExecutionPlan& step : execution_plan) {
      auto pnode = graph.GetNode(step.node_index);
      for (auto node_input : pnode->InputDefs()) {
        if (node_input->Exists())
          UseCount(node_input->Name())++;
      }
      for (auto node_output : pnode->OutputDefs())
        if (node_output->Exists())
          ProcessDef(node_output);
    }

    for (auto graph_output : graph.GetOutputs()) {
      UseCount(graph_output->Name())++;  // Models caller's usage post-inference; ensures it will not be reused.
    }
  }

  void ComputeReusePlan(const LotusIR::Graph& graph,
                        std::vector<SequentialExecutionPlan::NodeExecutionPlan>& execution_plan) {
    // Identify allocation/deallocation plan for every ml-value

    // inputs of the graph:
    // An input ml-value's data is owned by the caller (of InferenceSession::Run())
    // It must be allocated by the caller, and will not be reused during inference.
    for (auto graph_input : graph.GetInputs()) {
      auto input_index = index(graph_input->Name());
      SequentialExecutionPlan::AllocPlanPerValue& thisplan = AllocPlan(input_index);
      thisplan.alloc_kind = AllocKind::kPreExisting;
      thisplan.value_type = GetMLDataType(*graph_input);
    }

    auto& weights = graph.GetAllInitializedTensors();
    for (auto it = weights.begin(); it != weights.end(); ++it) {
      auto wt_index = index(it->first);
      SequentialExecutionPlan::AllocPlanPerValue& thisplan = AllocPlan(wt_index);
      thisplan.alloc_kind = AllocKind::kAllocateStatically;
      // Note: thisplan.value_type should already have been setup since every initializer must be an input
    }

    for (int program_counter = 0; program_counter < execution_plan.size(); ++program_counter) {
      SequentialExecutionPlan::NodeExecutionPlan step = execution_plan[program_counter];
      auto pnode = graph.GetNode(step.node_index);
      // determine allocation for outputs of pnode
      int output_arg_num = 0;
      for (auto node_output : pnode->OutputDefs()) {
        if (!node_output->Exists()) continue;
        auto current = index(node_output->Name());
        AllocPlan(current).value_type = GetMLDataType(*node_output);
        MLValueIndex reused;
        if (IsNonTensor(*node_output)) {
          // we do not try sharing-optimization for non-tensors
          AllocPlan(current).alloc_kind = AllocKind::kAllocate;
        } else if (FindReusableInput(*pnode, output_arg_num, &reused)) {
          // Reuse one of this node's input buffers as the output buffer (for in-place update)
          Reuse(reused, current);
        } else if (FindReusableTensor(*node_output, &reused)) {
          // Reuse an available (dead) buffer for this output
          Reuse(reused, current);
        } else {
          // otherwise: allocate a new buffer for this output
          AllocPlan(current).alloc_kind = AllocKind::kAllocate;
        }
        output_arg_num++;
      }
      // determine if inputs of *pnode can be freed:
      for (auto node_input : pnode->InputDefs()) {
        if (node_input->Exists()) {
          auto& sym = node_input->Name();
          auto original = Buffer(index(sym));
          if (0 == --UseCount(original))
            freelist_.push_front(FreeBufferInfo(original, program_counter));
        }
      }
      // determine if any outputs of *pnode are unused and can be freed:
      for (auto node_output : pnode->OutputDefs()) {
        if (node_output->Exists()) {
          auto& sym = node_output->Name();
          auto original = Buffer(index(sym));
          if (0 == UseCount(original))
            freelist_.push_front(FreeBufferInfo(original, program_counter));
        }
      }
    }
  }

  // Convert information in a freelist (about which ml-value becomes free when) into
  // a deallocation plan in the format required in an ExecutionPlan
  static void GenerateDeallocationPlan(const std::list<FreeBufferInfo>& freelist,
                                       SequentialExecutionPlan* plan) {
    // Store (indices of) ml-values to be freed in plan->to_be_freed
    // Set plan->execution_plan[n].free_from_index/free_to_index for every n that must free some ml-value.

    plan->to_be_freed.reserve(freelist.size());
    int prev_dealloc_point = -1;  // when >=0, this indicates previous n that contains deallocations
    int current = 0;              // current index into the to_be_freed vector

    // Copy all items from freelist to to_be_freed in reverse order
    for (auto it = freelist.rbegin(); it != freelist.rend(); ++it) {
      plan->to_be_freed.push_back(it->ml_value);
      //
      if (it->deallocate_point != prev_dealloc_point) {
        if (prev_dealloc_point >= 0)
          plan->execution_plan[prev_dealloc_point].free_to_index = current - 1;
        prev_dealloc_point = it->deallocate_point;
        plan->execution_plan[prev_dealloc_point].free_from_index = current;
      }
      current++;
    }
    if (prev_dealloc_point >= 0)
      plan->execution_plan[prev_dealloc_point].free_to_index = current - 1;
  }

  bool IsNonTensor(const LotusIR::NodeArg& nodearg) {
    // TODO: unclear why we should go through a string-representation of type
    auto ptype = nodearg.Type();
    auto& type_proto = onnx::Utils::DataTypeUtils::ToTypeProto(ptype);
    return !type_proto.has_tensor_type();
  }

 public:
  Status CreatePlan(const SessionState& session_state, const ISequentialPlannerContext& context, SequentialExecutionPlan* plan) {
    p_session_state_ = &session_state;
    p_context_ = &context;
    plan_ = plan;

    auto p_graph = p_session_state_->GetGraph();
    LotusIR::Graph* p_nonconst_graph = const_cast<LotusIR::Graph*>(p_graph);

    const std::vector<LotusIR::NodeIndex>* p_graph_nodes;
    LOTUS_RETURN_IF_ERROR(p_nonconst_graph->GetNodesInTopologicalOrder(&p_graph_nodes));

    auto num_ml_values = session_state.GetMaxMLValueIdx() + 1;

    Initialize(p_graph_nodes->size(), num_ml_values);

    // Determine execution order: we use the default topological sort order for now. We can later
    // explore more efficient orderings (from a memory usage perspective).
    for (auto n : *p_graph_nodes) {
      if (!(p_graph->IsSourceNode(n) || p_graph->IsSinkNode(n)))
        plan_->execution_plan.emplace_back(n);
    }

    // compute usecounts for all ml-values
    ComputeUseCounts(*p_graph, plan_->execution_plan);

    // determine sharing/reuse among ml-values
    ComputeReusePlan(*p_graph, plan_->execution_plan);

    // convert information in the freelist_ into a deallocation plan in required format
    GenerateDeallocationPlan(freelist_, plan_);

    return Status::OK();
  }
};

Status SequentialPlanner::CreatePlan(const SessionState& session_state, const ISequentialPlannerContext& context,
                                     SequentialExecutionPlan* plan) {
  PlannerImpl planner;
  return planner.CreatePlan(session_state, context, plan);
}

Status AllocationPlanner::CreatePlan(AllocationPlannerType allocation_planner_type,
                                     const SessionState& session_state,
                                     SequentialExecutionPlan* plan) {
  switch (allocation_planner_type) {
    case AllocationPlannerType::SEQUENTIAL_PLANNER: {
      return SequentialPlanner::CreatePlan(session_state, plan);
    }
    case AllocationPlannerType::SIMPLE_SEQUENTIAL_PLANNER: {
      return SimpleAllocationPlanner::CreatePlan(session_state, plan);
    }
    default:
      return Status(LOTUS, FAIL, "Invalid allocation planner type requested");
  }
}

}  // namespace Lotus

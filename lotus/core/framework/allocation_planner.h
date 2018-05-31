#pragma once

#include "core/common/status.h"
#include "core/framework/execution_provider.h"
#include "core/graph/graph.h"
// TODO: inference_session is included to bring in the AllocationPlannerType
// This is for testing only (not intended for production usage)
#include "core/framework/inference_session.h"

namespace Lotus {

// Every ml-value has a unique name and is assigned a unique integral number.
// While we use names at static-planning time, the goal is that at runtime
// (that is, at inference time), there is no need to refer to names, and only
// the integer index is used (e.g., to index into appropriate vectors in
// the ExecutionFrame).
typedef int MLValueIndex;
typedef std::string MLValueName;

// The ML-Values fall into the following categories with respect to their
// memory management:
//   - inference inputs: owned (allocated and freed) by caller, and is by
//     default read-only by the runtime.
//   - inference outputs: allocated by runtime, ownership transferred to
//     caller. TODO: Make sure this semantics is clear in InferenceSession API.
//   - weights (constant tensors): can be allocated once (statically), and
//     reused by all inference calls within an InferenceSession.
//   - tensor values: The lifetimes of these tensor-values are statically
//     determined, which is used for memory reuse/sharing optimizations. The
//     runtime allocates/frees these values at the right time (as determined
//     by the static allocation plan). Note that this is simplified since we
//     do not try to optimize for "slice" like ops, where we may be able to
//     conditionally reuse memory/data in some cases but not others.
//     Generalizing this is future work.

enum class AllocKind {
  kAllocate = 0,
  kReuse = 1,
  kPreExisting = 2,
  kAllocateStatically = 3,
};

class SessionState;

// SequentialExecutionPlan: This is the data that is produced by a static
// planner for a sequential execution, to be used by a SequentialExecutor.
struct SequentialExecutionPlan {
  // Allocation plan:
  // ExecutionFrame::GetOrCreateTensor() should use the following information
  // to decide whether to allocate a new buffer or reuse an existing buffer

  // AllocPlanPerValue: (a simplified form of AllocationPlanPerValue above)
  // Captures information required to allocate/reuse buffer for a ml-value
  struct AllocPlanPerValue {
    AllocKind alloc_kind;
    MLDataType value_type;
    AllocatorInfo location;
    // reused_buffer is valid only if alloc_kind == kReuse. It indicates
    // which MLValue's buffer must be reused for this MLValue.
    MLValueIndex reused_buffer;
    // if the value is used in async kernel, a fence object would be created
    // note the fence object would be shared between MLValues reusing the same buffer
    bool create_fence;

   public:
    AllocPlanPerValue() : alloc_kind(AllocKind::kAllocate),
                          value_type(nullptr),
                          location(CPU, kArenaAllocator),
                          reused_buffer(0),
                          create_fence(false) {}
  };

  // The following vector is indexed by MLValueIndex
  std::vector<AllocPlanPerValue> allocation_plan;

  // The following indicates the order in which nodes should be executed and the
  // ml-values to be free after each node's execution:

  // NodeExecutionPlan: represents execution data for a single node
  struct NodeExecutionPlan {
    // node to be executed;
    LotusIR::NodeIndex node_index;

    // ml-values to be freed after node execution:
    // for (auto i = free_from_index; i <= free_to_index; i++)
    //    free ml-value corresponding to ml-value-index to_be_freed[i]
    int free_from_index;
    int free_to_index;

    NodeExecutionPlan(LotusIR::NodeIndex index) : node_index(index), free_from_index(1), free_to_index(0) {}
  };

  // Execution_plan: represents the nodes in the sequential order to be executed
  std::vector<NodeExecutionPlan> execution_plan;

  // to_be_freed: vector elements represent indices of ml-values to be freed (as described above)
  std::vector<MLValueIndex> to_be_freed;
};

// ISequentialPlannerContext abstracts how the planner accesses information (such as inferred shape)
// to do the planning.
class ISequentialPlannerContext {
 public:
  virtual const TensorShapeProto* GetShape(const LotusIR::NodeArg& arg) const = 0;
};

class SequentialPlannerContext : public ISequentialPlannerContext {
 public:
  virtual const TensorShapeProto* GetShape(const LotusIR::NodeArg& arg) const override {
    return arg.Shape();
  }
};

class SequentialPlanner {
 public:
  // This API allows user to provide a custom planner context. Currently, this is used
  // primarily for testing.
  static Status CreatePlan(const SessionState& session_state,
                           const ISequentialPlannerContext& context,
                           SequentialExecutionPlan* plan);

  // This uses a standard planner context and is meant to be the primary API for creating a plan.
  static Status CreatePlan(const SessionState& session_state, SequentialExecutionPlan* plan) {
    SequentialPlannerContext context;
    return CreatePlan(session_state, context, plan);
  }
};

/*
SimpleAllocationPlanner is used to generate a default execution plan for test.
For allocation part, all the values will be set as kAllocate, except
weights. Weights will be set as kAllocateStatically.
For execution, it is just follow the topological order, and won't free
any values in the middle of execution.
*/
class SimpleAllocationPlanner {
 public:
  static Status CreatePlan(const SessionState& session_state,
                           SequentialExecutionPlan* plan);
};

class AllocationPlanner {
 public:
  static Status CreatePlan(AllocationPlannerType allocation_planner_type,
                           const SessionState& session_state,
                           SequentialExecutionPlan* plan);
};

}  // namespace Lotus

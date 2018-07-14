#pragma once

#include "core/common/status.h"
#include "core/framework/execution_provider.h"
#include "core/framework/alloc_kind.h"
#include "core/framework/allocator.h"
#include "core/framework/session_state.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/graph.h"
namespace onnx {
class TensorShapeProto;
}
namespace Lotus {

// ISequentialPlannerContext abstracts how the planner accesses information (such as inferred shape)
// to do the planning.
class ISequentialPlannerContext {
 public:
  virtual const onnx::TensorShapeProto* GetShape(const LotusIR::NodeArg& arg) const = 0;
};

class SequentialPlannerContext : public ISequentialPlannerContext {
 public:
  const onnx::TensorShapeProto* GetShape(const LotusIR::NodeArg& arg) const override {
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

class AllocationPlanner {
 public:
  static Status CreatePlan(const SessionState& session_state,
                           SequentialExecutionPlan* plan);
};

}  // namespace Lotus

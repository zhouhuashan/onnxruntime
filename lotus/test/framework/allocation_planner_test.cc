#include "core/framework/allocation_planner.h"
#include <string>
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
namespace Test {

namespace GraphBuilder {

// Type: a wrapper to build a TypeProto
struct Type {
  TypeProto value;

  // construct a float-tensor-type
  Type() {
    value.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  }

  // construct a float-tensor-type with given constant dimensions
  Type(std::initializer_list<int> dims) {
    value.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    auto p_shape = value.mutable_tensor_type()->mutable_shape();
    for (auto d : dims) {
      auto dim = p_shape->add_dim();
      dim->set_dim_value(d);
    }
  }

  // construct a float-tensor-type with given symbolic dimensions
  Type(std::initializer_list<string> symbolic_dims) {
    value.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    auto p_shape = value.mutable_tensor_type()->mutable_shape();
    for (auto d : symbolic_dims) {
      auto dim = p_shape->add_dim();
      dim->set_dim_param(d);
    }
  }
};

// Arg: A wrapper to build a NodeArg
struct Arg {
  LotusIR::NodeArg value;

  Arg(const std::string& name, const TypeProto* p_arg_type) : value(name, p_arg_type) {}

  Arg(const std::string& name, const Type& type) : value(name, &type.value) {}
};

void AddNames(SessionState& state, std::initializer_list<string> names) {
  int count = 0;
  for (auto& name : names) {
    state.AddMLValueNameIdx(name, count++);
  }
}

class NodeCounter {
 private:
  static int node_count_;

 public:
  static int Next() { return ++node_count_; }
};

int NodeCounter::node_count_ = 0;

struct UnaryNode {
  std::vector<LotusIR::NodeArg*> input_args;
  std::vector<LotusIR::NodeArg*> output_args;
  LotusIR::Node* p_node;

  UnaryNode(LotusIR::Graph* graph, const std::string& op,
            LotusIR::NodeArg* p_input_arg, LotusIR::NodeArg* p_output_arg)
      : input_args({p_input_arg}), output_args({p_output_arg}) {
    int num = NodeCounter::Next();
    p_node = graph->AddNode("node" + std::to_string(num), op, "test op", input_args, output_args);
  }

  UnaryNode(LotusIR::Graph* graph, LotusIR::NodeArg* p_input_arg, LotusIR::NodeArg* p_output_arg)
      : UnaryNode(graph, "Transpose", p_input_arg, p_output_arg) {}

  UnaryNode(LotusIR::Graph* graph, const std::string& op, Arg* p_input_arg, Arg* p_output_arg)
      : UnaryNode(graph, op, &p_input_arg->value, &p_output_arg->value) {}

  UnaryNode(LotusIR::Graph* graph, Arg* p_input_arg, Arg* p_output_arg)
      : UnaryNode(graph, "Transpose", p_input_arg, p_output_arg) {}
};

class DummyOpKernel : public OpKernel {
 public:
  DummyOpKernel(const OpKernelInfo& p) : OpKernel(p) {}
  Status Compute(OpKernelContext* context) const {
    UNUSED_PARAMETER(context);
    return Status::OK();
  }
  Status ComputeAsync(OpKernelContext* context, DoneCallback done) const {
    UNUSED_PARAMETER(context);
    return Status::OK();
  }
};

}  // namespace GraphBuilder

using namespace GraphBuilder;

class AllocationPlanTestUtility {
 public:
  static void CheckAllocationKind(const SequentialExecutionPlan& plan, std::vector<AllocKind>& expected) {
    ASSERT_EQ(plan.allocation_plan.size(), expected.size()) << "Allocation plan of wrong size";
    for (int i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(plan.allocation_plan[i].alloc_kind, expected[i]) << "Error in allocation kind at position " << i;
    }
  }

  static void CheckToBeFreed(const SequentialExecutionPlan& plan, const std::vector<MLValueIndex>& expected) {
    ASSERT_EQ(plan.to_be_freed.size(), expected.size()) << "Allocation plan's to_be_freed of wrong size";
    for (int i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(plan.to_be_freed[i], expected[i]) << "Error in to_be_freed at position " << i;
    }
  }

  static void CheckFreedAtEachStep(const SequentialExecutionPlan& plan, const std::vector<int>& expected_num_freed) {
    ASSERT_EQ(plan.execution_plan.size(), expected_num_freed.size()) << "Execution plan is of wrong size";
    int start = 0;
    for (int i = 0; i < expected_num_freed.size(); ++i) {
      if (expected_num_freed[i] > 0) {
        EXPECT_EQ(plan.execution_plan[i].free_from_index, start) << "Error in free_from_index at position " << i;
        EXPECT_EQ(plan.execution_plan[i].free_to_index, start + expected_num_freed[i] - 1)
            << "Error in free_to_index at position " << i;
        start = start + expected_num_freed[i];
      } else {
        // "free_from_index > free_to_index" indicates nothing is to be freed
        EXPECT_GT(plan.execution_plan[i].free_from_index, plan.execution_plan[i].free_to_index);
      }
    }
  }

  static void BasicIntegrityCheck(const SequentialExecutionPlan& plan, int num_ml_values) {
    // Sanity checks for plan.to_be_freed
    std::unordered_set<MLValueIndex> freed;
    for (MLValueIndex index : plan.to_be_freed) {
      // Every index should be in the valid range [0, num_ml_values-1]
      EXPECT_GE(index, 0);
      EXPECT_LT(index, num_ml_values);
      // An index should not be freed more than once
      EXPECT_EQ(freed.count(index), 0) << "MLValue " << index << " freed multiple times";
      freed.insert(index);
    }
    // Check the free-index information for every execution step: they should cover the
    // range [0, plan.to_be_freed.size()-1] properly.
    int next_free_index = 0;
    int max_free_index = ((int)plan.to_be_freed.size()) - 1;
    for (const SequentialExecutionPlan::NodeExecutionPlan& step : plan.execution_plan) {
      if (step.free_from_index <= step.free_to_index) {
        EXPECT_EQ(step.free_from_index, next_free_index);
        EXPECT_LE(step.free_to_index, max_free_index);
        next_free_index = step.free_to_index + 1;
      }  // else nothing needs to be freed in this step
    }
  }
};

TEST(AllocationPlannerTest, ChainTest) {
  LotusIR::Model model("test");
  LotusIR::Graph* graph = model.MainGraph();

  Type type1{100, 50};
  Type type2{1};

  LotusIR::NodeArg w_def("W", &type2.value);
  LotusIR::NodeArg x_def("X", &type1.value);
  LotusIR::NodeArg b_def("B", &type1.value);
  LotusIR::NodeArg y_def("Y", &type1.value);

  onnx::TensorProto tensor;
  tensor.add_dims(1);
  tensor.add_float_data(1.0f);
  tensor.set_data_type(TensorProto_DataType_FLOAT);
  tensor.set_name("W");
  graph->AddInitializedTensor(tensor);

  UnaryNode node1(graph, &w_def, &x_def);
  UnaryNode node2(graph, &x_def, &b_def);
  UnaryNode node3(graph, &b_def, &y_def);

  SessionState state;
  state.SetGraph(graph);
  int b_idx = 2;  // w_idx = 0, x_idx = 1, y_idx = 3;
  AddNames(state, {"W", "X", "B", "Y"});

  SequentialExecutionPlan plan;
  auto status = SequentialPlanner::CreatePlan(state, &plan);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // Expected plan:
  //   W: kAllocateStatically; X: kAllocate; B: kAllocate; Y: kReuse (X); post-node3: free(B); X is returned output
  std::vector<AllocKind> expected_alloc(
      {AllocKind::kAllocateStatically, AllocKind::kAllocate, AllocKind::kAllocate, AllocKind::kReuse});
  AllocationPlanTestUtility::CheckAllocationKind(plan, expected_alloc);

  // Note: Y (which reuses X) is treated as graph output and should not be freed
  std::vector<MLValueIndex> expected_to_be_freed({b_idx});
  AllocationPlanTestUtility::CheckToBeFreed(plan, expected_to_be_freed);

  std::vector<int> expected_num_freed({0, 0, 1});
  AllocationPlanTestUtility::CheckFreedAtEachStep(plan, expected_num_freed);
}

/* InputOutputTest: Test that:
(a) All inputs are classified as kPreExisting,
(b) All outputs are classified as kAllocate (in this example),
(c) Neither input nor outputs are freed.
*/
TEST(AllocationPlannerTest, InputOutputTest) {
  LotusIR::Model model("test");
  LotusIR::Graph* graph = model.MainGraph();

  Type type;

  Arg X1("X1", type);
  Arg X2("X2", type);
  Arg Y1("Y1", type);
  Arg Y2("Y2", type);

  UnaryNode node1(graph, &X1, &Y1);
  UnaryNode node2(graph, &X2, &Y2);

  SessionState state;
  state.SetGraph(graph);
  AddNames(state, {"X1", "X2", "Y1", "Y2"});

  auto kernel_def1 = KernelDefBuilder("Transpose").Build();
  AllocatorInfo allocator_info("CPUAllocator", AllocatorType::kArenaAllocator);

  OpKernelInfo info1(*(node1.p_node), allocator_info, *kernel_def1);
  state.AddKernel(node1.p_node->Index(), std::make_unique<DummyOpKernel>(info1));

  OpKernelInfo info2(*(node2.p_node), allocator_info, *kernel_def1);
  state.AddKernel(node2.p_node->Index(), std::make_unique<DummyOpKernel>(info2));

  SequentialExecutionPlan plan;
  auto status = SequentialPlanner::CreatePlan(state, &plan);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // X1: kPreExisting, X2: kPreExisting, Y1: kAllocate, Y2: kAllocate
  std::vector<AllocKind> expected_alloc(
      {AllocKind::kPreExisting, AllocKind::kPreExisting, AllocKind::kAllocate, AllocKind::kAllocate});
  AllocationPlanTestUtility::CheckAllocationKind(plan, expected_alloc);

  // Nothing should be freed (since they are either inputs or outputs)
  std::vector<MLValueIndex> expected_to_be_freed({});
  AllocationPlanTestUtility::CheckToBeFreed(plan, expected_to_be_freed);

  std::vector<int> expected_num_freed({0, 0});
  AllocationPlanTestUtility::CheckFreedAtEachStep(plan, expected_num_freed);
}

// InPlaceTest: Check that we reuse when Inplace allows us to.

TEST(AllocationPlannerTest, InPlaceTest) {
  LotusIR::Model model("test");
  LotusIR::Graph* graph = model.MainGraph();

  Type type1{"M", "N"};

  Arg X1("X1", type1);
  Arg X2("X2", type1);
  Arg X3("X3", type1);
  Arg X4("X4", type1);

  UnaryNode node1(graph, "Transpose", &X1, &X2);  // no in-place operator; X1: input; X2: temporary
  UnaryNode node2(graph, "Clip", &X2, &X3);       // may-in-place operator; X3: temporary
  UnaryNode node3(graph, "Transpose", &X3, &X4);  // no in-place operator; X4: output

  SessionState state;
  state.SetGraph(graph);
  AddNames(state, {"X1", "X2", "X3", "X4"});

  auto kernel_def1 = KernelDefBuilder("Transpose").Build();
  auto kernel_def2 = KernelDefBuilder("Clip").MayInplace(0, 0).Build();

  AllocatorInfo allocator_info("CPUAllocator", AllocatorType::kArenaAllocator);

  OpKernelInfo info1(*(node1.p_node), allocator_info, *kernel_def1);
  state.AddKernel(node1.p_node->Index(), std::make_unique<DummyOpKernel>(info1));

  OpKernelInfo info2(*(node2.p_node), allocator_info, *kernel_def2);
  state.AddKernel(node2.p_node->Index(), std::make_unique<DummyOpKernel>(info2));

  OpKernelInfo info3(*(node3.p_node), allocator_info, *kernel_def1);
  state.AddKernel(node3.p_node->Index(), std::make_unique<DummyOpKernel>(info3));

  SequentialExecutionPlan plan;
  auto status = SequentialPlanner::CreatePlan(state, &plan);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // X1: kPreExisting, X2: kAllocate, X3: kReuse, X4: kAllocate
  std::vector<AllocKind> expected_alloc(
      {AllocKind::kPreExisting, AllocKind::kAllocate, AllocKind::kReuse, AllocKind::kAllocate});
  AllocationPlanTestUtility::CheckAllocationKind(plan, expected_alloc);

  // X2 should be freed
  std::vector<MLValueIndex> expected_to_be_freed({1});
  AllocationPlanTestUtility::CheckToBeFreed(plan, expected_to_be_freed);

  // X2 should be freed after last node
  std::vector<int> expected_num_freed({0, 0, 1});
  AllocationPlanTestUtility::CheckFreedAtEachStep(plan, expected_num_freed);
}

/* TODO: We can't yet test for size/shape mismatch; need some more infrastructure to
    enable this.

// InPlaceSizeMismatchTest: Check that Inplace reuse is not allowed when sizes don't match.
// Also tests reuse of disjoint lifetime tensors.
TEST(AllocationPlannerTest, InPlaceSizeMismatchTest) {
  LotusIR::Model model("test");
  LotusIR::Graph* graph = model.MainGraph();

  Type type1{"M", "N"};
  Type type2{"M", "K"};

  Arg X1("X1", type1);
  Arg X2("X2", type1);
  Arg X3("X3", type2);
  Arg X4("X4", type1);

  UnaryNode node1(graph, "Transpose", &X1, &X2);  // no in-place operator; X1: input; X2: temporary
  UnaryNode node2(graph, "Clip", &X2, &X3);       // may-in-place operator; X3: temporary
  UnaryNode node3(graph, "Transpose", &X3, &X4);  // no in-place operator; X4: output

  SessionState state;
  state.SetGraph(graph);
  AddNames(state, {"X1", "X2", "X3", "X4"});

  auto kernel_def1 = KernelDefBuilder("Transpose").Build();
  auto kernel_def2 = KernelDefBuilder("Clip").MayInplace(0, 0).Build();

  AllocatorInfo allocator_info("CPUAllocator", AllocatorType::kArenaAllocator);

  OpKernelInfo info1(*(node1.p_node), allocator_info, *kernel_def1);
  state.AddKernel(node1.p_node->Index(), std::make_unique<DummyOpKernel>(info1));

  OpKernelInfo info2(*(node2.p_node), allocator_info, *kernel_def2);
  state.AddKernel(node2.p_node->Index(), std::make_unique<DummyOpKernel>(info2));

  OpKernelInfo info3(*(node3.p_node), allocator_info, *kernel_def1);
  state.AddKernel(node3.p_node->Index(), std::make_unique<DummyOpKernel>(info3));

  SequentialExecutionPlan plan;
  auto status = SequentialPlanner::CreatePlan(state, &plan);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // X1: kPreExisting, X2: kAllocate, X3: kAllocate, X4: kReuse (X2)
  std::vector<AllocKind> expected_alloc(
      {AllocKind::kPreExisting, AllocKind::kAllocate, AllocKind::kAllocate, AllocKind::kReuse});
  AllocationPlanTestUtility::CheckAllocationKind(plan, expected_alloc);

  // X3 should be freed
  std::vector<MLValueIndex> expected_to_be_freed({2});
  AllocationPlanTestUtility::CheckToBeFreed(plan, expected_to_be_freed);

  // X3 should be freed after last node
  std::vector<int> expected_num_freed({0, 0, 1});
  AllocationPlanTestUtility::CheckFreedAtEachStep(plan, expected_num_freed);
}

*/

}  // namespace Test
}  // namespace Lotus

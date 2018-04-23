
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "core/framework/op_kernel.h"
#include "test/framework/model_builder_utils.h"
#include "core/framework/allocation_planner.h"

namespace Lotus {
namespace Test {

namespace ModelBuilder {

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

}  // namespace ModelBuilder

using namespace ModelBuilder;

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

  static void BasicIntegrityCheck(const SequentialExecutionPlan& plan, size_t num_ml_values) {
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

typedef std::unordered_map<const LotusIR::NodeArg*, TensorShapeProto*> ShapeMap;

class SequentialPlannerTestContext : public ISequentialPlannerContext {
 public:
  SequentialPlannerTestContext(ShapeMap* shape_map) : shape_map_(shape_map) {}

  virtual TensorShapeProto* GetShape(const LotusIR::NodeArg& arg) const override {
    auto iter = shape_map_->find(&arg);
    return (shape_map_->end() != iter) ? iter->second : nullptr;
  }

 private:
  ShapeMap* shape_map_;
};

class PlannerTest : public ::testing::Test {
 protected:
  LotusIR::Model model_;
  LotusIR::Graph* graph_;

  // some standard components used to build test-cases:
  Type float_type_;

  std::unique_ptr<Lotus::KernelDef> std_kernel_;       // a unary kernel with no-aliasing and no-in-place
  std::unique_ptr<Lotus::KernelDef> in_place_kernel_;  // a unary kernel with in-place

  std::unordered_map<string, LotusIR::NodeArg*> name_to_arg_;
  std::vector<UnaryNode*> nodes_;
  std::vector<OpKernelInfo*> op_kernel_infos_;
  std::vector<std::pair<LotusIR::Node*, KernelDef&>> kernel_bindings_;
  SessionState state_;
  AllocatorInfo allocator_info_;
  ShapeMap shape_map_;

 public:
  PlannerTest() : model_("test"), allocator_info_("CPUAllocator", AllocatorType::kArenaAllocator) {
    graph_ = model_.MainGraph();
    std_kernel_ = KernelDefBuilder("Transpose").Build();
    in_place_kernel_ = KernelDefBuilder("Clip").MayInplace(0, 0).Build();
  }

  ~PlannerTest() {
    for (auto& pair : name_to_arg_)
      delete pair.second;
    for (auto p_node : nodes_)
      delete p_node;
    for (auto p_op_kernel_info : op_kernel_infos_)
      delete p_op_kernel_info;
  }

  LotusIR::NodeArg* Arg(const std::string& name) {
    auto iter = name_to_arg_.find(name);
    if (name_to_arg_.end() != iter) return iter->second;
    auto arg = new LotusIR::NodeArg(name, &float_type_.value);
    name_to_arg_[name] = arg;
    return arg;
  }

  LotusIR::Node* AddNode(Lotus::KernelDef& kernel_def, std::string& input, std::string& output) {
    auto t = new UnaryNode(graph_, kernel_def.OpName(), Arg(input), Arg(output));
    nodes_.push_back(t);
    kernel_bindings_.emplace_back(t->p_node, kernel_def);
    return t->p_node;
  }

  LotusIR::Node* AddNormalNode(std::string& input, std::string& output) {
    return AddNode(*std_kernel_, input, output);
  }

  LotusIR::Node* AddInplaceNode(std::string& input, std::string& output) {
    return AddNode(*in_place_kernel_, input, output);
  }

  void BindKernel(LotusIR::Node* p_node, Lotus::KernelDef& kernel_def) {
    auto t = new OpKernelInfo(*p_node, allocator_info_, kernel_def, nullptr);
    op_kernel_infos_.push_back(t);
    state_.AddKernel(p_node->Index(), std::make_unique<DummyOpKernel>(*t));
  }

  void SetShape(std::string& name, TensorShapeProto* shape) {
    shape_map_[Arg(name)] = shape;
  }

  void SetShape(std::initializer_list<std::pair<std::string&, TensorShapeProto*>> shapes) {
    for (auto& pair : shapes) {
      SetShape(pair.first, pair.second);
    }
  }

  SequentialExecutionPlan plan_;

  void CreatePlan() {
    EXPECT_EQ(graph_->Resolve(), Status::OK());
    state_.SetGraph(graph_);

    int count = 0;
    for (auto& pair : name_to_arg_) {
      state_.AddMLValueNameIdx(pair.first, count++);
    }

    for (auto& binding : kernel_bindings_) {
      BindKernel(binding.first, binding.second);
    }

    SequentialPlannerTestContext test_context(&shape_map_);
    auto status = SequentialPlanner::CreatePlan(state_, test_context, &plan_);
    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    AllocationPlanTestUtility::BasicIntegrityCheck(plan_, name_to_arg_.size());
  }

  int index(const std::string& name) {
    int indx;
    state_.GetMLValueIdx(name, &indx);
    return indx;
  }

  void CheckAllocKind(const std::string& name, AllocKind kind) {
    EXPECT_EQ(plan_.allocation_plan[index(name)].alloc_kind, kind) << "Error in allocation kind for " << name;
  }

  void CheckFreed(int step_number, std::initializer_list<std::string> freed_items) {
    // create set and check equality
    std::unordered_set<int> expected;
    for (auto& name : freed_items) {
      expected.insert(index(name));
    }
    std::unordered_set<int> plan_result;
    auto& step_plan = plan_.execution_plan[step_number];
    for (int i = step_plan.free_from_index; i <= step_plan.free_to_index; ++i)
      plan_result.insert(plan_.to_be_freed[i]);
    EXPECT_EQ(plan_result, expected) << "Freed items incorrect for step " << step_number;
  }
};

TEST_F(PlannerTest, ChainTest) {
  // tensor variables:
  std::string W("W"), X("X"), B("B"), Y("Y");

  // graph structure:

  onnx::TensorProto tensor;
  tensor.add_dims(1);
  tensor.add_float_data(1.0f);
  tensor.set_data_type(TensorProto_DataType_FLOAT);
  tensor.set_name("W");
  graph_->AddInitializedTensor(tensor);

  AddNormalNode(W, X);
  AddNormalNode(X, B);
  AddNormalNode(B, Y);

  // simulate shape-inference results:
  Shape shape1{50, 100};
  auto shape = &shape1.value;
  SetShape({{X, shape}, {B, shape}, {Y, shape}});

  CreatePlan();

  // Expected plan:
  //   W: kAllocateStatically; X: kAllocate; B: kAllocate; Y: kReuse (X); post-node3: free(B); X is returned output
  CheckAllocKind(W, AllocKind::kAllocateStatically);
  CheckAllocKind(X, AllocKind::kAllocate);
  CheckAllocKind(B, AllocKind::kAllocate);
  CheckAllocKind(Y, AllocKind::kReuse);

  // Note: Y (which reuses X) is treated as graph output and should not be freed
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {"B"});
}

/* InputOutputTest: Test that:
(a) All inputs are classified as kPreExisting,
(b) All outputs are classified as kAllocate (in this example),
(c) Neither input nor outputs are freed.
*/
TEST_F(PlannerTest, InputOutputTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), Y1("Y1"), Y2("Y2");

  // graph structure:
  AddNormalNode(X1, Y1);
  AddNormalNode(X2, Y2);

  // simulate no shape-inference:

  CreatePlan();

  // X1: kPreExisting, X2: kPreExisting, Y1: kAllocate, Y2: kAllocate
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kPreExisting);
  CheckAllocKind(Y1, AllocKind::kAllocate);
  CheckAllocKind(Y2, AllocKind::kAllocate);

  // Nothing should be freed (since they are either inputs or outputs)
  CheckFreed(0, {});
  CheckFreed(1, {});
}

// InPlaceTest: Check that we reuse when Inplace allows us to.

TEST_F(PlannerTest, InPlaceTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4");

  // graph structure:
  AddNormalNode(X1, X2);   // no in-place operator; X1: input; X2: temporary
  AddInplaceNode(X2, X3);  // may-in-place operator; X3: temporary
  AddNormalNode(X3, X4);   // no in-place operator; X4: output

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}, {X4, shape}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocate);
  CheckAllocKind(X3, AllocKind::kReuse);
  CheckAllocKind(X4, AllocKind::kAllocate);

  // check each ml-value is freed at appropriate step
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {X2});
}

// InPlaceSizeMismatchTest: Check that Inplace reuse is not allowed when sizes don't match.
// Also tests reuse of disjoint lifetime tensors.
TEST_F(PlannerTest, InPlaceSizeMismatchTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4");

  // graph structure:
  AddNormalNode(X1, X2);   // no in-place operator; X1: input; X2: temporary
  AddInplaceNode(X2, X3);  // may-in-place operator; X3: temporary
  AddNormalNode(X3, X4);   // no in-place operator; X4: output

  // simulate shape-inference results:
  Shape shape1w{"M", "N"};
  auto shape1 = &shape1w.value;
  Shape shape2w{"M", "K"};
  auto shape2 = &shape2w.value;
  SetShape({{X1, shape1}, {X2, shape1}, {X3, shape2}, {X4, shape1}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocate);
  CheckAllocKind(X3, AllocKind::kAllocate);
  CheckAllocKind(X4, AllocKind::kReuse);

  // check each ml-value is freed at appropriate step
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {X3});
}

}  // namespace Test
}  // namespace Lotus

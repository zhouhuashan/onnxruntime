#include "core/framework/allocation_planner.h"
#include <string>
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace Test {

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

  UnaryNode(LotusIR::Graph* graph,
            NodeArg* p_inputArg,
            NodeArg* p_outputArg) : input_args({p_inputArg}), output_args({p_outputArg}) {
    int num = NodeCounter::Next();
    p_node = graph->AddNode("node" + std::to_string(num), "Clip", "test op", input_args, output_args);
  }
};

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
    ASSERT_EQ(plan.execution_plan.size(), expected_num_freed.size()) << "Allocation plan's execution plan of wrong size";
    int start = 0;
    for (int i = 0; i < expected_num_freed.size(); ++i) {
      if (expected_num_freed[i] > 0) {
        EXPECT_EQ(plan.execution_plan[i].free_from_index, start) << "Error in free_from_index at position " << i;
        EXPECT_EQ(plan.execution_plan[i].free_to_index, start + expected_num_freed[i] - 1) << "Error in free_to_index at position " << i;
        start = start + expected_num_freed[i];
      } else {
        // "free_from_index > free_to_index" indicates nothing is to be freed
        EXPECT_GT(plan.execution_plan[i].free_from_index, plan.execution_plan[i].free_to_index);
      }
    }
  }
};

TEST(AllocationPlannerTest, ChainNoShapeTest) {
  LotusIR::Model model("test");
  LotusIR::Graph* graph = model.MainGraph();

  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  TypeProto weight_float;
  weight_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto dim = weight_float.mutable_tensor_type()->mutable_shape()->add_dim();
  dim->set_dim_value(1);

  LotusIR::NodeArg w_def("W", &weight_float);
  LotusIR::NodeArg x_def("X", &tensor_float);
  LotusIR::NodeArg b_def("B", &tensor_float);
  LotusIR::NodeArg y_def("Y", &tensor_float);

  onnx::TensorProto tensor;
  tensor.add_dims(1);
  tensor.add_float_data(1.0f);
  tensor.set_data_type(TensorProto_DataType_FLOAT);
  tensor.set_name("W");
  graph->AddInitializedTensor(tensor);

  // auto node = graph->GetNode(graph->NumberOfNodes() - 1);
  UnaryNode node1(graph, &w_def, &x_def);
  UnaryNode node2(graph, &x_def, &b_def);
  UnaryNode node3(graph, &b_def, &y_def);

  SessionState state;
  state.SetGraph(graph);
  int w_idx = 0, x_idx = 1, b_idx = 2, y_idx = 3;
  state.AddMLValueNameIdx("W", w_idx);
  state.AddMLValueNameIdx("X", x_idx);
  state.AddMLValueNameIdx("B", b_idx);
  state.AddMLValueNameIdx("Y", y_idx);

  SequentialExecutionPlan plan;
  auto status = SequentialPlanner::CreatePlan(state, &plan);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::vector<AllocKind> expected_alloc({AllocKind::kAllocateStatically, AllocKind::kAllocate, AllocKind::kAllocate, AllocKind::kAllocate});
  AllocationPlanTestUtility::CheckAllocationKind(plan, expected_alloc);

  // Note: Y is treated as graph output and should not be freed
  std::vector<MLValueIndex> expected_to_be_freed({x_idx, b_idx});
  AllocationPlanTestUtility::CheckToBeFreed(plan, expected_to_be_freed);

  std::vector<int> expected_num_freed({0, 1, 1});
  AllocationPlanTestUtility::CheckFreedAtEachStep(plan, expected_num_freed);

  // ASSERT_EQ(plan.execution_plan.size(), 1) << "Execution plan has wrong size";

  // EXPECT_GT(plan.execution_plan[0].free_from_index,
  // EXPECT_EQ(plan.execution_plan[0].node_index, node->Index());
  // EXPECT_GT(plan.execution_plan[0].free_from_index, plan.execution_plan[0].free_to_index);
}
}  // namespace Test
}  // namespace Lotus

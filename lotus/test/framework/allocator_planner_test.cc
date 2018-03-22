#include "core/framework/allocation_planner.h"
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace Test {
TEST(AllocationPlannerTest, DummyPlannerTest) {
  LotusIR::Model model("test");
  LotusIR::Graph* graph = model.MainGraph();

  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  TypeProto weight_float;
  weight_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto dim = weight_float.mutable_tensor_type()->mutable_shape()->add_dim();
  dim->set_dim_value(1);

  LotusIR::NodeArg x_def("X", &tensor_float),
      w_def("W", &weight_float),
      b_def("B", &tensor_float),
      output_def("Y", &tensor_float);
  std::vector<LotusIR::NodeArg*> input_defs{&x_def, &w_def, &b_def};
  std::vector<LotusIR::NodeArg*> output_defs{&output_def};
  graph->AddNode("node1", "Gemm", "test op", input_defs, output_defs);
  auto node = graph->GetNode(graph->NumberOfNodes() - 1);

  onnx::TensorProto tensor;
  tensor.add_dims(1);
  tensor.add_float_data(1.0f);
  tensor.set_data_type(TensorProto_DataType_FLOAT);
  tensor.set_name("W");
  graph->AddInitializedTensor(tensor);

  SessionState state;
  state.SetGraph(graph);
  state.AddMLValueNameIdx("X", 0);
  state.AddMLValueNameIdx("W", 1);
  state.AddMLValueNameIdx("B", 2);
  state.AddMLValueNameIdx("Y", 3);

  SequentialExecutionPlan dummy_plan;
  auto status = SimpleAllocationPlanner::CreatePlan(state, &dummy_plan);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(dummy_plan.allocation_plan.size(), 4);
  EXPECT_EQ(dummy_plan.allocation_plan[0].alloc_kind, AllocKind::kAllocate);
  EXPECT_EQ(dummy_plan.allocation_plan[1].alloc_kind, AllocKind::kAllocateStatically);
  EXPECT_EQ(dummy_plan.allocation_plan[2].alloc_kind, AllocKind::kAllocate);
  EXPECT_EQ(dummy_plan.allocation_plan[3].alloc_kind, AllocKind::kAllocate);

  EXPECT_EQ(dummy_plan.execution_plan.size(), 1);
  EXPECT_EQ(dummy_plan.execution_plan[0].node_index, node->Index());
  EXPECT_GT(dummy_plan.execution_plan[0].free_from_index, dummy_plan.execution_plan[0].free_to_index);
}
}  // namespace Test
}  // namespace Lotus

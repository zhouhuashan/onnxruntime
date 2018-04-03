#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace Test {
typedef std::vector<LotusIR::NodeArg*> ArgMap;

std::shared_ptr<LotusIR::Model> DummyGraphWithClip() {
  auto model = std::make_shared<LotusIR::Model>("test");
  LotusIR::Graph* graph = model->MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  LotusIR::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);

  graph->AddNode("node1", "Clip", "clip operator", ArgMap{&input_def}, ArgMap{&output_def});
  return model;
}

TEST(ExecutionFrameTest, TensorAllocationTest) {
  LotusIR::Model model("test");
  LotusIR::Graph* graph = model.MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  LotusIR::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);

  graph->AddNode("node1", "Clip", "Clip operator", ArgMap{&input_def}, ArgMap{&output_def});
  LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);

  AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::kArenaAllocator);

  SessionState state;
  state.SetGraph(graph);
  state.AddMLValueNameIdx("X", 0);
  state.AddMLValueNameIdx("Y", 1);
  ExecutionFrame frame(std::unordered_map<std::string, MLValue>{},
                       std::vector<std::string>{},
                       state);

  int start_index = frame.GetFirstArgIndex(node->Index());
  EXPECT_EQ(start_index, 0);

  TensorShape shape(std::vector<int64_t>{2, 3});
  auto status = frame.AllocateTensorWithSelfOwnBuffer(start_index, DataTypeImpl::GetType<float>(),
                                                      AllocatorManager::Instance().GetArena(CPU).Info(), shape);
  EXPECT_TRUE(status.IsOK());

  auto tensor = frame.GetMutableValue<Tensor>(0);
  EXPECT_TRUE(tensor);
  EXPECT_EQ(tensor->Shape(), shape);
  EXPECT_EQ(tensor->DataType(), DataTypeImpl::GetType<float>());

  //test share memory from tensor
  TensorShape shape2(std::vector<int64_t>{3, 2});
  status = frame.AllocateTensorWithPreAllocateBuffer(
      start_index + 1,
      tensor->MutableData<float>(),
      DataTypeImpl::GetType<float>(),
      tensor->Location(),
      shape2);
  EXPECT_TRUE(status.IsOK());

  auto tensor2 = frame.GetValue<Tensor>(1);
  EXPECT_TRUE(tensor2);
  EXPECT_EQ(tensor2->Shape(), shape2);
  EXPECT_EQ(tensor2->Data<float>(), tensor->Data<float>());
}

TEST(ExecutionFrameTest, FeedInDataTest) {
  LotusIR::Model model("test");
  LotusIR::Graph* graph = model.MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  LotusIR::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);

  graph->AddNode("node1", "Clip", "Clip operator", ArgMap{&input_def}, ArgMap{&output_def});

  auto& cpu_allocator = AllocatorManager::Instance().GetArena(CPU);
  auto element_type = DataTypeImpl::GetType<float>();
  TensorShape shape({3, 2});
  void* buffer = cpu_allocator.Alloc(element_type->Size() * shape.Size());
  //create fake ml value with owned buffer.
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(
      element_type,
      shape,
      buffer,
      cpu_allocator.Info(),
      &cpu_allocator);
  MLValue value;
  value.Init(p_tensor.release(),
             DataTypeImpl::GetType<Tensor>(),
             DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  SessionState state;
  state.SetGraph(graph);
  state.AddMLValueNameIdx("X", 0);
  state.AddMLValueNameIdx("Y", 1);
  ExecutionFrame frame(std::unordered_map<std::string, MLValue>{{"X", value}},
                       std::vector<std::string>{},
                       state);

  auto tensor = frame.GetMutableValue<Tensor>(0);
  EXPECT_TRUE(tensor);
  EXPECT_EQ(tensor->Shape(), shape);
  EXPECT_EQ(tensor->DataType(), DataTypeImpl::GetType<float>());
  EXPECT_EQ(tensor->MutableData<float>(), buffer);
}
}  // namespace Test
}  // namespace Lotus

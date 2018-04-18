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
  vector<MLValue> outputs;
  ExecutionFrame frame(std::unordered_map<std::string, MLValue>{},
                       std::vector<std::string>{},
                       outputs,
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
  vector<MLValue> outputs;
  ExecutionFrame frame(std::unordered_map<std::string, MLValue>{{"X", value}},
                       std::vector<std::string>{},
                       outputs,
                       state);

  auto tensor = frame.GetMutableValue<Tensor>(0);
  EXPECT_TRUE(tensor);
  EXPECT_EQ(tensor->Shape(), shape);
  EXPECT_EQ(tensor->DataType(), DataTypeImpl::GetType<float>());
  EXPECT_EQ(tensor->MutableData<float>(), buffer);
}
template <typename T>
void CreateMLValue(IAllocator* alloc,
	const std::vector<int64_t>& dims,
	const std::vector<T>& value,
	MLValue* p_mlvalue) {
	TensorShape shape(dims);
	auto location = alloc->Info();
	auto element_type = DataTypeImpl::GetType<T>();
	void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
	if (value.size() > 0) {
		memcpy(buffer, &value[0], element_type->Size() * shape.Size());
	}

	std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
		shape,
		buffer,
		location,
		alloc);
	p_mlvalue->Init(p_tensor.release(),
		DataTypeImpl::GetType<Tensor>(),
		DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

TEST(ExecutionFrameTest, MemPatternTest) {
	LotusIR::Model model("test");
	LotusIR::Graph* graph = model.MainGraph();
	TypeProto tensor_float;
	tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
	LotusIR::NodeArg input_def1("X1", &tensor_float),
		input_def2("X2", &tensor_float),
		input_def3("X3", &tensor_float),
		gemm1_out_def("T1", &tensor_float),
		gemm2_out_def("T2", &tensor_float),
		clip_out_def("T3", &tensor_float)
		;

	graph->AddNode("node1", "MatMul", "gemm1", ArgMap{ &input_def1, &input_def2 }, ArgMap{ &gemm1_out_def });

	graph->AddNode("node2", "MatMul", "gemm2", ArgMap{ &gemm1_out_def, &input_def3 }, ArgMap{ &gemm2_out_def });

	graph->AddNode("node3", "Clip", "clip1", ArgMap{ &gemm2_out_def }, ArgMap{ &clip_out_def });

	auto status = graph->Resolve();
	EXPECT_TRUE(status.IsOK());
	//1. prepare input
	auto& cpu_allocator = AllocatorManager::Instance().GetArena(CPU);
	MLValue v1, v2, v3;
	CreateMLValue<float>(&cpu_allocator, 
		std::vector<int64_t>{1, 2}, 
		std::vector<float>{1.0f, 1.0f}, &v1);
	CreateMLValue<float>(&cpu_allocator, 
		std::vector<int64_t>{2, 2}, 
		std::vector<float>(4, 1.0f), &v2);
	CreateMLValue<float>(&cpu_allocator, 
		std::vector<int64_t>{2, 3}, 
		std::vector<float>(6, 1.0f), &v3);

	SessionState state;
	state.SetGraph(graph);
	state.AddMLValueNameIdx("X1", 0);
	state.AddMLValueNameIdx("X2", 1);
	state.AddMLValueNameIdx("X3", 2);
	state.AddMLValueNameIdx("T1", 3);
	state.AddMLValueNameIdx("T2", 4);
	state.AddMLValueNameIdx("T3", 5);

	std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan = std::make_unique<SequentialExecutionPlan>();
	// TODO below line is for testing only. In production use SequentialPlanner::CreatePlan()
	status = AllocationPlanner::CreatePlan(AllocationPlannerType::SIMPLE_SEQUENTIAL_PLANNER,
		state,
		p_seq_exec_plan.get());
	EXPECT_TRUE(status.IsOK());

	state.SetExecutionPlan(std::move(p_seq_exec_plan));

	vector<MLValue> outputs;
	ExecutionFrame frame(std::unordered_map<std::string, MLValue>{ {"X1", v1}, { "X2", v2 }, {"X3", v3}},
		std::vector<std::string>{"T3"},
		outputs,
		state);

	status = frame.AllocateMLValueTensorSelfOwnBuffer(3,
		DataTypeImpl::GetType<float>(),
		cpu_allocator.Info(),
		TensorShape(std::vector<int64_t>{2, 2}));
	EXPECT_TRUE(status.IsOK());

	status = frame.AllocateMLValueTensorSelfOwnBuffer(4,
		DataTypeImpl::GetType<float>(),
		cpu_allocator.Info(),
		TensorShape(std::vector<int64_t>{2, 3}));
	EXPECT_TRUE(status.IsOK());

	status = frame.AllocateMLValueTensorSelfOwnBuffer(5,
		DataTypeImpl::GetType<float>(),
		cpu_allocator.Info(),
		TensorShape(std::vector<int64_t>{2, 3}));
	EXPECT_TRUE(status.IsOK());

	MemoryPatternGroup pattern;
	status = frame.GeneratePatterns(&pattern);
	EXPECT_TRUE(status.IsOK());

	EXPECT_EQ(pattern.patterns.size(), pattern.locations.size());
	EXPECT_EQ(pattern.patterns.size(), 1);
	auto p = pattern.GetPatterns(cpu_allocator.Info());
	EXPECT_EQ(p->peak_size(), sizeof(float) * (4 + 6));
	EXPECT_EQ(p->GetBlock(3)->offset_, 0);
	EXPECT_EQ(p->GetBlock(4)->offset_, sizeof(float) * 4);

}
}  // namespace Test
}  // namespace Lotus

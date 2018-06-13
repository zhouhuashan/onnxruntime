#include "core/framework/inference_session.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <thread>

#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/framework/tensorprotoutils.h"

#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "../test_utils.h"
#include "gtest/gtest.h"

using namespace std;
using namespace Lotus::Logging;

namespace Lotus {
namespace Test {

typedef std::vector<LotusIR::NodeArg*> ArgMap;

TEST(CUDAFenceTests, PartOnCPU) {
  std::unique_ptr<LotusIR::Model> model = std::make_unique<LotusIR::Model>("test");
  LotusIR::Graph* graph = model->MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  LotusIR::NodeArg x1_def("X1", &tensor_float);
  LotusIR::NodeArg y_def("Y", &tensor_float);
  LotusIR::NodeArg z_def("Z", &tensor_float);
  LotusIR::NodeArg out_def("Out", &tensor_float);

  onnx::TensorProto weight;
  weight.add_dims(2);
  weight.add_dims(2);
  weight.set_data_type(TensorProto_DataType_FLOAT);
  weight.add_float_data(-1);
  weight.add_float_data(2);
  weight.add_float_data(3);
  weight.add_float_data(-4);
  weight.set_name("W");
  graph->AddInitializedTensor(weight);
  auto& w_def = graph->GetOrCreateNodeArg("W", &tensor_float);

  auto p_node = graph->AddNode("node1", "MatMul", "MatMul operator", ArgMap{&w_def, &x1_def}, ArgMap{&y_def});
  p_node->SetExecutionProviderType(LotusIR::kCudaExecutionProvider);
  p_node = graph->AddNode("node2", "Add", "Add operator", ArgMap{&y_def, &w_def}, ArgMap{&z_def});
  p_node->SetExecutionProviderType(LotusIR::kCpuExecutionProvider);
  p_node = graph->AddNode("node3", "Add", "Add operator", ArgMap{&y_def, &z_def}, ArgMap{&out_def});
  p_node->SetExecutionProviderType(LotusIR::kCpuExecutionProvider);

  // add and then delete a node to test node iteration against nullptr
  p_node = graph->AddNode("node_to_delete", "Add", "Add operator", ArgMap{&y_def, &z_def}, ArgMap{&out_def});
  graph->RemoveNode(p_node->Index());

  EXPECT_TRUE(graph->Resolve().IsOK());

  auto cpu_allocator = TestCPUExecutionProvider()->GetAllocator();
  auto element_type = DataTypeImpl::GetType<float>();
  TensorShape shape({2, 2});
  float data[4] = {-1, 2, 3, -4};
  void* buffer = cpu_allocator->Alloc(element_type->Size() * shape.Size());
  memcpy(buffer, data, sizeof(data));

  //create fake ml value with owned buffer.
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(
      element_type,
      shape,
      buffer,
      cpu_allocator->Info(),
      cpu_allocator);
  MLValue value;
  value.Init(p_tensor.release(),
             DataTypeImpl::GetType<Tensor>(),
             DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  SessionOptions so;
  InferenceSession session(so);
  session.Load(std::move(model));
  CUDAExecutionProviderInfo xp_info;
  session.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(xp_info));
  EXPECT_TRUE(session.Initialize().IsOK());

  size_t num_copy_nodes = 0;
  for (auto& p : graph->Nodes())
    num_copy_nodes += (p.OpType().substr(0, 6) == "Memcpy");
  EXPECT_TRUE(2 == num_copy_nodes);

  vector<MLValue> outputs;
  session.Run(std::unordered_map<std::string, MLValue>{{"X1", value}},
              std::vector<std::string>{"Out"},
              &outputs);
  EXPECT_TRUE(1 == outputs.size());
  const Tensor& output = outputs[0].Get<Tensor>();
  EXPECT_EQ(output.Shape(), shape);
  EXPECT_EQ(output.DataType(), DataTypeImpl::GetType<float>());

  float expected_output[4] = {13.0f, -18.0f, -27.0f, 40.0f};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(output.Data<float>()[i], expected_output[i]);
  }
}

}  // namespace Test
}  // namespace Lotus

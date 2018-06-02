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
  LotusIR::Model model("test");
  LotusIR::Graph* graph = model.MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  LotusIR::NodeArg x0_def("X0", &tensor_float);
  LotusIR::NodeArg x1_def("X1", &tensor_float);
  LotusIR::NodeArg y_def("Y", &tensor_float);
  LotusIR::NodeArg z_def("Z", &tensor_float);

  auto p_node = graph->AddNode("node1", "MatMul", "MatMul operator", ArgMap{&x0_def, &x1_def}, ArgMap{&y_def});
  p_node->SetExecutionProviderType(LotusIR::kCudaExecutionProvider);
  p_node = graph->AddNode("node2", "Relu", "Relu operator", ArgMap{&y_def}, ArgMap{&z_def});
  p_node->SetExecutionProviderType(LotusIR::kCpuExecutionProvider);

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

  SessionState state;
  state.SetGraph(graph);
  state.AddMLValueNameIdx("X0", 0);
  state.AddMLValueNameIdx("X1", 1);
  state.AddMLValueNameIdx("Y", 2);
  state.AddMLValueNameIdx("Z", 3);
  CUDAExecutionProviderInfo xp_info;
  state.AddExecutionProvider(LotusIR::kCudaExecutionProvider, std::make_unique<CUDAExecutionProvider>(xp_info));
  vector<MLValue> outputs;

  Tensor* output = nullptr;
  {
    ExecutionFrame frame(std::unordered_map<std::string, MLValue>{{"X0", value}, {"X1", value}},
                         std::vector<std::string>{},
                         outputs,
                         state);

    output = frame.GetMutableValue<Tensor>(0);
  }
  EXPECT_TRUE(output);
  EXPECT_EQ(output->Shape(), shape);
  EXPECT_EQ(output->DataType(), DataTypeImpl::GetType<float>());

  float expected_output[4] = {7.0f, 0.0f, 0.0f, 22.0f};
  EXPECT_EQ(output->MutableData<float>(), buffer);
}

}  // namespace Test
}  // namespace Lotus

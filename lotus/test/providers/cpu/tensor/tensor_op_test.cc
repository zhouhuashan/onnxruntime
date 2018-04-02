#include "core/providers/cpu/tensor/reshape.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

typedef std::vector<LotusIR::NodeArg*> ArgMap;
TEST(TensorOpTest, Reshape) {
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  LotusIR::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);
  std::vector<LotusIR::NodeArg*> input_defs{&input_def};
  std::vector<LotusIR::NodeArg*> output_defs{&output_def};
  CREATE_NODE("Reshape", input_defs, output_defs);

  EXPECT_TRUE(node->AddAttribute("shape", std::vector<int64_t>{-1, 0, 2}));

  AllocatorInfo allocator_info("CPUAllocator", AllocatorType::kArenaAllocator);
  KernelDef kernel_def;
  OpKernelInfo info(*node, allocator_info, kernel_def);
  Reshape<float> kernel(info);

  std::vector<float> input_vals(6, 1.0f);
  std::vector<int64_t> input_shape({2, 3});
  std::vector<int64_t> expected_shape({1, 3, 2});

  SessionState state;
  state.SetGraph(graph);
  state.AddMLValueNameIdx("X", 0);
  state.AddMLValueNameIdx("Y", 1);
  auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(state, {{"X", MLValue()}}, {"Y"});
  auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, input_shape, &input_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, expected_shape);
  EXPECT_TRUE(status.IsOK());

  OpKernelContext kernel_ctx(frame.get(), static_cast<OpKernel*>(&kernel), DefaultLoggingManager().DefaultLogger());
  kernel.Compute(&kernel_ctx);
  auto res = kernel_ctx.Output(0, TensorShape(expected_shape));
  for (int i = 0; i < input_vals.size(); ++i) {
    EXPECT_EQ(input_vals[i], res->Data<float>()[i]);
  }
}
}  // namespace Test
}  // namespace Lotus

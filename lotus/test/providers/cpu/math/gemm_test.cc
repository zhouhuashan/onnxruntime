#include "core/providers/cpu/math/clip.h"
#include "core/providers/cpu/math/gemm.h"
#include "core/providers/cpu/math/matmul.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {
static const TypeProto_Set s_typeProto_float{TensorProto_DataType_FLOAT};

TEST(MathOpTest, GemmNoTrans) {
  LotusIR::NodeArg x_def("X", &s_typeProto_float),
      w_def("W", &s_typeProto_float),
      b_def("B", &s_typeProto_float),
      output_def("Y", &s_typeProto_float);
  std::vector<LotusIR::NodeArg*> input_defs{&x_def, &w_def, &b_def};
  std::vector<LotusIR::NodeArg*> output_defs{&output_def};
  CREATE_NODE("Gemm", input_defs, output_defs);

  EXPECT_TRUE(node->AddAttribute("transA", (int64_t)0));
  EXPECT_TRUE(node->AddAttribute("transB", (int64_t)0));
  EXPECT_TRUE(node->AddAttribute("broadcast", (int64_t)0));
  EXPECT_TRUE(node->AddAttribute("alpha", 1.0f));
  EXPECT_TRUE(node->AddAttribute("beta", 1.0f));

  AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
  KernelDef kernel_def;
  OpKernelInfo info(*node, allocator_info, kernel_def);
  Gemm<float, float, float, float> kernel(info);

  std::vector<float> x_vals{1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<int64_t> x_dims{2, 4};
  std::vector<float> y_vals(12, 1.0f);
  std::vector<int64_t> y_dims{4, 3};
  std::vector<float> b_vals(6, 1.0f);
  std::vector<int64_t> b_dims{2, 3};
  std::vector<float> expected_vals{11.0f, 11.0f, 11.0f, -9.0f, -9.0f, -9.0f};
  std::vector<int64_t> expected_dims{2, 3};

  SessionState state;
  state.SetGraph(graph);
  SetupState(state, input_defs, output_defs);

  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(input_defs, output_defs, feeds, output_names);

  auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(state, feeds, output_names);
  auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, x_dims, &x_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthInput<float>(*node, 1, frame, y_dims, &y_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthInput<float>(*node, 2, frame, b_dims, &b_vals);
  EXPECT_TRUE(status.IsOK());

  status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, expected_dims);
  EXPECT_TRUE(status.IsOK());

  OpKernelContext kernel_ctx(frame.get(), static_cast<OpKernel*>(&kernel), DefaultLoggingManager().DefaultLogger());
  kernel.compute(&kernel_ctx);
  auto output = kernel_ctx.output(0, TensorShape(expected_dims));
  const float* res = output->data<float>();

  for (int i = 0; i < expected_vals.size(); ++i) {
    EXPECT_EQ(expected_vals[i], res[i]);
  }
}

TEST(MathOpTest, GemmBroadcast) {
  LotusIR::NodeArg x_def("X", &s_typeProto_float),
      w_def("W", &s_typeProto_float),
      b_def("B", &s_typeProto_float),
      output_def("Y", &s_typeProto_float);
  std::vector<LotusIR::NodeArg*> input_defs{&x_def, &w_def, &b_def};
  std::vector<LotusIR::NodeArg*> output_defs{&output_def};
  CREATE_NODE("Gemm", input_defs, output_defs);

  EXPECT_TRUE(node->AddAttribute("transA", (int64_t)0));
  EXPECT_TRUE(node->AddAttribute("transB", (int64_t)0));
  EXPECT_TRUE(node->AddAttribute("broadcast", (int64_t)1));
  EXPECT_TRUE(node->AddAttribute("alpha", 1.0f));
  EXPECT_TRUE(node->AddAttribute("beta", 1.0f));

  AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
  KernelDef kernel_def;
  OpKernelInfo info(*node, allocator_info, kernel_def);
  Gemm<float, float, float, float> kernel(info);

  std::vector<float> x_vals{1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<int64_t> x_dims{2, 4};
  std::vector<float> y_vals(12, 1.0f);
  std::vector<int64_t> y_dims{4, 3};
  std::vector<float> b_vals(3, 1.0f);
  std::vector<int64_t> b_dims{3};
  std::vector<float> expected_vals{11.0f, 11.0f, 11.0f, -9.0f, -9.0f, -9.0f};
  std::vector<int64_t> expected_dims{2, 3};

  SessionState state;
  state.SetGraph(graph);
  SetupState(state, input_defs, output_defs);

  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(input_defs, output_defs, feeds, output_names);

  auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(state, feeds, output_names);
  auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, x_dims, &x_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthInput<float>(*node, 1, frame, y_dims, &y_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthInput<float>(*node, 2, frame, b_dims, &b_vals);
  EXPECT_TRUE(status.IsOK());

  status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, expected_dims);
  EXPECT_TRUE(status.IsOK());

  OpKernelContext kernel_ctx(frame.get(), static_cast<OpKernel*>(&kernel), DefaultLoggingManager().DefaultLogger());
  kernel.compute(&kernel_ctx);
  auto output = kernel_ctx.output(0, TensorShape(expected_dims));
  const float* res = output->data<float>();

  for (int i = 0; i < expected_vals.size(); ++i) {
    EXPECT_EQ(expected_vals[i], res[i]);
  }
}

TEST(MathOpTest, GemmTrans) {
  LotusIR::NodeArg x_def("X", &s_typeProto_float),
      w_def("W", &s_typeProto_float),
      b_def("B", &s_typeProto_float),
      output_def("Y", &s_typeProto_float);
  std::vector<LotusIR::NodeArg*> input_defs{&x_def, &w_def, &b_def};
  std::vector<LotusIR::NodeArg*> output_defs{&output_def};
  CREATE_NODE("Gemm", input_defs, output_defs);

  EXPECT_TRUE(node->AddAttribute("transA", (int64_t)1));
  EXPECT_TRUE(node->AddAttribute("transB", (int64_t)1));
  EXPECT_TRUE(node->AddAttribute("broadcast", (int64_t)1));
  EXPECT_TRUE(node->AddAttribute("alpha", 1.0f));
  EXPECT_TRUE(node->AddAttribute("beta", 1.0f));

  AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
  KernelDef kernel_def;
  OpKernelInfo info(*node, allocator_info, kernel_def);
  Gemm<float, float, float, float> kernel(info);

  std::vector<float> x_vals{1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
  std::vector<int64_t> x_dims{4, 2};
  std::vector<float> y_vals(12, 1.0f);
  std::vector<int64_t> y_dims{3, 4};
  std::vector<float> b_vals(3, 1.0f);
  std::vector<int64_t> b_dims{3};
  std::vector<float> expected_vals{11.0f, 11.0f, 11.0f, -9.0f, -9.0f, -9.0f};
  std::vector<int64_t> expected_dims{2, 3};

  SessionState state;
  state.SetGraph(graph);
  SetupState(state, input_defs, output_defs);

  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(input_defs, output_defs, feeds, output_names);

  auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(state, feeds, output_names);
  auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, x_dims, &x_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthInput<float>(*node, 1, frame, y_dims, &y_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthInput<float>(*node, 2, frame, b_dims, &b_vals);
  EXPECT_TRUE(status.IsOK());

  status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, expected_dims);
  EXPECT_TRUE(status.IsOK());

  OpKernelContext kernel_ctx(frame.get(), static_cast<OpKernel*>(&kernel), DefaultLoggingManager().DefaultLogger());
  kernel.compute(&kernel_ctx);
  auto output = kernel_ctx.output(0, TensorShape(expected_dims));
  const float* res = output->data<float>();

  for (int i = 0; i < expected_vals.size(); ++i) {
    EXPECT_EQ(expected_vals[i], res[i]);
  }
}

TEST(MathOpTest, GemmAlphaBeta) {
  LotusIR::NodeArg x_def("X", &s_typeProto_float),
      w_def("W", &s_typeProto_float),
      b_def("B", &s_typeProto_float),
      output_def("Y", &s_typeProto_float);
  std::vector<LotusIR::NodeArg*> input_defs{&x_def, &w_def, &b_def};
  std::vector<LotusIR::NodeArg*> output_defs{&output_def};
  CREATE_NODE("Gemm", input_defs, output_defs);

  EXPECT_TRUE(node->AddAttribute("transA", (int64_t)0));
  EXPECT_TRUE(node->AddAttribute("transB", (int64_t)0));
  EXPECT_TRUE(node->AddAttribute("broadcast", (int64_t)1));
  EXPECT_TRUE(node->AddAttribute("alpha", 0.5f));
  EXPECT_TRUE(node->AddAttribute("beta", 2.0f));

  AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
  KernelDef kernel_def;
  OpKernelInfo info(*node, allocator_info, kernel_def);
  Gemm<float, float, float, float> kernel(info);

  std::vector<float> x_vals{1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<int64_t> x_dims{2, 4};
  std::vector<float> y_vals(12, 1.0f);
  std::vector<int64_t> y_dims{4, 3};
  std::vector<float> b_vals(3, 1.0f);
  std::vector<int64_t> b_dims{3};
  std::vector<float> expected_vals{7.0f, 7.0f, 7.0f, -3.0f, -3.0f, -3.0f};
  std::vector<int64_t> expected_dims{2, 3};

  SessionState state;
  state.SetGraph(graph);
  SetupState(state, input_defs, output_defs);

  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(input_defs, output_defs, feeds, output_names);

  auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(state, feeds, output_names);
  auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, x_dims, &x_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthInput<float>(*node, 1, frame, y_dims, &y_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthInput<float>(*node, 2, frame, b_dims, &b_vals);
  EXPECT_TRUE(status.IsOK());

  status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, expected_dims);
  EXPECT_TRUE(status.IsOK());

  OpKernelContext kernel_ctx(frame.get(), static_cast<OpKernel*>(&kernel), DefaultLoggingManager().DefaultLogger());
  kernel.compute(&kernel_ctx);
  auto output = kernel_ctx.output(0, TensorShape(expected_dims));
  const float* res = output->data<float>();

  for (int i = 0; i < expected_vals.size(); ++i) {
    EXPECT_EQ(expected_vals[i], res[i]);
  }
}

TEST(MathOpTest, GemmNaN) {
  LotusIR::NodeArg x_def("X", &s_typeProto_float),
      w_def("W", &s_typeProto_float),
      b_def("B", &s_typeProto_float),
      output_def("Y", &s_typeProto_float);
  std::vector<LotusIR::NodeArg*> input_defs{&x_def, &w_def, &b_def};
  std::vector<LotusIR::NodeArg*> output_defs{&output_def};
  CREATE_NODE("Gemm", input_defs, output_defs);

  EXPECT_TRUE(node->AddAttribute("transA", (int64_t)0));
  EXPECT_TRUE(node->AddAttribute("transB", (int64_t)0));
  EXPECT_TRUE(node->AddAttribute("broadcast", (int64_t)0));
  EXPECT_TRUE(node->AddAttribute("alpha", 1.0f));
  EXPECT_TRUE(node->AddAttribute("beta", 0.0f));

  AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
  KernelDef kernel_def;
  OpKernelInfo info(*node, allocator_info, kernel_def);
  Gemm<float, float, float, float> kernel(info);

  std::vector<float> x_vals{1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<int64_t> x_dims{2, 4};
  std::vector<float> y_vals(12, 1.0f);
  std::vector<int64_t> y_dims{4, 3};
  std::vector<float> b_vals(6, 1.0f);
  std::vector<int64_t> b_dims{2, 3};
  std::vector<float> expected_vals{10.0f, 10.0f, 10.0f, -10.0f, -10.0f, -10.0f};
  std::vector<int64_t> expected_dims{2, 3};

  SessionState state;
  state.SetGraph(graph);
  SetupState(state, input_defs, output_defs);

  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(input_defs, output_defs, feeds, output_names);

  auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(state, feeds, output_names);
  auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, x_dims, &x_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthInput<float>(*node, 1, frame, y_dims, &y_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthInput<float>(*node, 2, frame, b_dims, &b_vals);
  EXPECT_TRUE(status.IsOK());

  status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, expected_dims);
  EXPECT_TRUE(status.IsOK());

  OpKernelContext kernel_ctx(frame.get(), static_cast<OpKernel*>(&kernel), DefaultLoggingManager().DefaultLogger());
  //set Y to Nan to making sure NaN does not propagate when beta == 0
  float nan = static_cast<float>(std::nan("1"));
  float* out_buffer = kernel_ctx.output(0, TensorShape(expected_dims))->mutable_data<float>();
  for (int i = 0; i < expected_vals.size(); ++i) {
    out_buffer[i] = nan;
  }

  kernel.compute(&kernel_ctx);
  auto output = kernel_ctx.output(0, TensorShape(expected_dims));
  const float* res = output->data<float>();

  for (int i = 0; i < expected_vals.size(); ++i) {
    EXPECT_EQ(expected_vals[i], res[i]);
  }
}

}  // namespace Test
}  // namespace Lotus

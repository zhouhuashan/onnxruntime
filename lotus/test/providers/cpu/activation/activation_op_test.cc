#include "core/providers/cpu/activation/activations.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {
static const TypeProto_Set s_typeProto_float{TensorProto_DataType_FLOAT};

template <class Op>
void TestUnaryElementwiseOp(std::vector<float>& input_vals, std::function<float(float)> expected_func, const std::unordered_map<std::string, float> attribs = {}, float abs_error = 0) {
  LotusIR::NodeArg input_def("X", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  CREATE_NODE(Op::TypeTraits(), {&input_def}, {&output_def});

  for (auto attr : attribs)
    node->AddAttribute(attr.first, attr.second);

  AllocatorInfo allocator_info("CPUAllocator", AllocatorType::kArenaAllocator);
  KernelDef kernel_def;
  OpKernelInfo info(*node, allocator_info, kernel_def);
  Op kernel(info);
  SessionState state;
  state.SetGraph(graph);
  state.AddMLValueNameIdx("X", 0);
  state.AddMLValueNameIdx("Y", 1);
  auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(state, {{"X", MLValue()}}, {"Y"});

  std::vector<int64_t> dims{(int64_t)input_vals.size()};

  std::vector<float> expected_vals;
  for (const auto& iv : input_vals)
    expected_vals.push_back(expected_func(iv));

  auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, dims, &input_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, dims);
  EXPECT_TRUE(status.IsOK());

  OpKernelContext ctx(frame.get(), static_cast<OpKernel*>(&kernel), DefaultLoggingManager().DefaultLogger());
  kernel.Compute(&ctx);
  auto Output = ctx.Output(0, TensorShape(dims));
  const float* res = Output->Data<float>();

  for (int i = 0; i < expected_vals.size(); ++i) {
    if (abs_error == 0)
      EXPECT_EQ(expected_vals[i], res[i]);
    else
      EXPECT_NEAR(expected_vals[i], res[i], abs_error);
  }
}

std::vector<float> input_vals = {
    -1.0f, 0, 1.0f,                                              // normal input values for activation
    FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,                        // min, denorm, -denorm
    FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity()};  // max, -max, inf

TEST(ActivationOpTest, Sigmoid) {
  TestUnaryElementwiseOp<Sigmoid<float>>(
      input_vals,
      [](float x) {
        auto y = 1.f / (1.f + std::exp(-abs(x)));  // safe sigmoid
        y = x > 0 ? y : 1 - y;
        return y;
      },
      {},
      FLT_EPSILON);
}

TEST(ActivationOpTest, Tanh) {
  TestUnaryElementwiseOp<Tanh<float>>(
      input_vals,
      [](float x) { return std::tanh(x); },
      {},
      FLT_MIN);
}

TEST(ActivationOpTest, Relu) {
  TestUnaryElementwiseOp<Relu<float>>(
      input_vals,
      [](float x) { return std::max(x, 0.0f); });
}

TEST(ActivationOpTest, Elu) {
  float alpha = 0.1f;
  TestUnaryElementwiseOp<Elu<float>>(
      input_vals,
      [alpha](float x) { return (x >= 0) ? x : alpha * (exp(x) - 1); },
      {{"alpha", alpha}});
}

TEST(ActivationOpTest, LeakyRelu) {
  float alpha = 0.1f;
  TestUnaryElementwiseOp<LeakyRelu<float>>(
      input_vals,
      [alpha](float x) { return (x >= 0) ? x : alpha * x; },
      {{"alpha", alpha}});
}

TEST(ActivationOpTest, ThresholdedRelu) {
  float alpha = 0.1f;
  TestUnaryElementwiseOp<ThresholdedRelu<float>>(
      input_vals,
      [alpha](float x) { return (x >= alpha) ? x : 0; },
      {{"alpha", alpha}});
}

}  // namespace Test
}  // namespace Lotus

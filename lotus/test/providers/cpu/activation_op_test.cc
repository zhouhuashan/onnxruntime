#include "core/providers/cpu/activation/relu.h"
#include "core/providers/cpu/activation/sigmoid.h"
#include "core/providers/cpu/activation/tanh.h"
#include "gtest/gtest.h"
#include "test/test_utils.h"

namespace Lotus {
namespace Test {
template <class Op>
void TestUnaryElementwiseOp(std::vector<float>& input_vals, std::function<float(float)> expected_func, float abs_error = 0) {
  LotusIR::NodeArg input_def("X", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  CREATE_NODE(Op::TypeTraits(), {&input_def}, {&output_def});

  AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
  KernelDef kernel_def;
  OpKernelInfo info(*node, allocator_info, kernel_def);

  Op kernel(info);
  SessionState state;
  state.Init(graph);
  auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(state);

  std::vector<int64_t> dims{(int64_t)input_vals.size()};

  std::vector<float> expected_vals;
  for (const auto& iv : input_vals)
    expected_vals.push_back(expected_func(iv));

  auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, dims, &input_vals);
  EXPECT_TRUE(status.IsOK());
  status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, dims);
  EXPECT_TRUE(status.IsOK());

  OpKernelContext ctx(frame.get(), static_cast<OpKernel*>(&kernel));
  kernel.compute(&ctx);
  auto output = ctx.output(0, TensorShape(dims));
  const float* res = output->data<float>();

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
      FLT_EPSILON);
}

TEST(ActivationOpTest, Tanh) {
  TestUnaryElementwiseOp<Tanh<float>>(
      input_vals,
      [](float x) { return std::tanh(x); },
      FLT_MIN);
}

TEST(ActivationOpTest, Relu) {
  TestUnaryElementwiseOp<Relu<float>>(
      input_vals,
      [](float f) { return std::max(f, 0.0f); });
}
}  // namespace Test
}  // namespace Lotus

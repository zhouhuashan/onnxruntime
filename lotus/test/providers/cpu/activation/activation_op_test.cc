#include "core/providers/cpu/activation/activations.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

void TestUnaryElementwiseOp(const char* szOp, std::vector<float>& input_vals, std::function<float(float)> expected_func, const std::unordered_map<std::string, float> attribs = {}) {
  OpTester test(szOp);

  for (auto attr : attribs)
    test.AddAttribute(attr.first, attr.second);

  std::vector<int64_t> dims{(int64_t)input_vals.size()};

  std::vector<float> expected_vals;
  for (const auto& iv : input_vals)
    expected_vals.push_back(expected_func(iv));

  test.AddInput<float>("X", dims, input_vals);
  test.AddOutput<float>("Y", dims, expected_vals);
  test.Run();
}

std::vector<float> input_vals = {
    -1.0f, 0, 1.0f,                                              // normal input values for activation
    FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,                        // min, denorm, -denorm
    FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity()};  // max, -max, inf

TEST(ActivationOpTest, Sigmoid) {
  TestUnaryElementwiseOp("Sigmoid",
                         input_vals,
                         [](float x) {
                           auto y = 1.f / (1.f + std::exp(-abs(x)));  // safe sigmoid
                           y = x > 0 ? y : 1 - y;
                           return y;
                         });
}

TEST(ActivationOpTest, Tanh) {
  TestUnaryElementwiseOp("Tanh",
                         input_vals,
                         [](float x) { return std::tanh(x); });
}

TEST(ActivationOpTest, Relu) {
  TestUnaryElementwiseOp("Relu",
                         input_vals,
                         [](float x) { return std::max(x, 0.0f); });
}

TEST(ActivationOpTest, Elu) {
  float alpha = 0.1f;
  TestUnaryElementwiseOp("Elu",
                         input_vals,
                         [alpha](float x) { return (x >= 0) ? x : alpha * (exp(x) - 1); },
                         {{"alpha", alpha}});
}

TEST(ActivationOpTest, LeakyRelu) {
  float alpha = 0.1f;
  TestUnaryElementwiseOp("LeakyRelu",
                         input_vals,
                         [alpha](float x) { return (x >= 0) ? x : alpha * x; },
                         {{"alpha", alpha}});
}

TEST(ActivationOpTest, ThresholdedRelu) {
  float alpha = 0.1f;
  TestUnaryElementwiseOp("ThresholdedRelu",
                         input_vals,
                         [alpha](float x) { return (x >= alpha) ? x : 0; },
                         {{"alpha", alpha}});
}

}  // namespace Test
}  // namespace Lotus

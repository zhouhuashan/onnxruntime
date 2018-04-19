#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(MLOpTest, ScalerOp) {
  OpTester test("Scaler", LotusIR::kMLDomain);
  vector<float> scale{3.f, -4.f, 3.0f};
  vector<float> offset{4.8f, -0.5f, 77.0f};
  test.AddAttribute("scale", scale);
  test.AddAttribute("offset", offset);
  vector<float> input{0.8f, -0.5f, 0.0f, 0.8f, 1.0f, 1.0f};
  vector<int64_t> dims{2, 3};

  // prepare expected output
  vector<float> expected_output;
  for (size_t i = 0; i < input.size(); ++i) {
    expected_output.push_back((input[i] - offset[i % dims[1]]) * scale[i % dims[1]]);
  }

  test.AddInput<float>("X", dims, input);
  test.AddOutput<float>("Y", dims, expected_output);
  test.Run();
}

TEST(MLOpTest, ScalerOpScaleOffsetSize1) {
  OpTester test("Scaler", LotusIR::kMLDomain);
  vector<float> scale{3.f};
  vector<float> offset{4.8f};
  test.AddAttribute("scale", scale);
  test.AddAttribute("offset", offset);
  vector<float> input{0.8f, -0.5f, 0.0f, 0.8f, 1.0f, 1.0f};
  vector<int64_t> dims{2, 3};

  // prepare expected output
  vector<float> expected_output;
  for (size_t i = 0; i < input.size(); ++i) {
    expected_output.push_back((input[i] - offset[0]) * scale[0]);
  }

  test.AddInput<float>("X", dims, input);
  test.AddOutput<float>("Y", dims, expected_output);
  test.Run();
}

}  // namespace Test
}  // namespace Lotus

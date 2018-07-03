#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(MLOpTest, BinarizerOp) {
  OpTester test("Binarizer", LotusIR::kMLDomain);
  float threshold = 0.3f;
  test.AddAttribute("threshold", threshold);
  vector<float> input{0.8f, -0.5f, 0.2f, 0.8f, -1.0f, 0.1f};

  // setup expected output
  vector<float> expected_output;
  for (auto& elem : input) {
    expected_output.push_back(elem > threshold ? 1.f : 0.f);
  }

  vector<int64_t> dims{2, 3};
  test.AddInput<float>("X", dims, input);
  test.AddOutput<float>("Y", dims, expected_output);
  test.Run();
}

}  // namespace Test
}  // namespace Lotus

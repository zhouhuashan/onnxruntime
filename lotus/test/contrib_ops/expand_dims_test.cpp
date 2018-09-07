#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(MLOpTest, ExpandDims_0) {
  OpTester test("ExpandDims", 1, LotusIR::kMSDomain);
  test.AddInput<float>("X", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("axis", {}, {-1});
  test.AddOutput<float>("Y", {2, 3, 1}, std::vector<float>(6, 1.0f));
  test.Run();
}

TEST(MLOpTest, ExpandDims_1) {
  OpTester test("ExpandDims", 1, LotusIR::kMSDomain);
  test.AddInput<float>("X", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("axis", {}, {1});
  test.AddOutput<float>("Y", {2, 1, 3}, std::vector<float>(6, 1.0f));
  test.Run();
}

}  // namespace Test
}  // namespace Lotus

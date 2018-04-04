#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

typedef std::vector<LotusIR::NodeArg*> ArgMap;
TEST(TensorOpTest, Reshape) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2});
  test.AddOutput<float>("reshaped", {1, 3, 2}, std::vector<float>(6, 1.0f));
  test.Run();
}
}  // namespace Test
}  // namespace Lotus

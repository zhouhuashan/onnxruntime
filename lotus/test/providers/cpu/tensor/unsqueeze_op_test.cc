#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(TensorOpTest, Unsqueeze_1) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {2, 1, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Unsqueeze_2) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{0, 4});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {1, 2, 3, 4, 1}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Unsqueeze_3) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{2, 1, 0});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Unsqueeze_Duplicate) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{2, 1, 0, 2});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run(OpTester::ExpectResult::kExpectFailure, "'axes' has a duplicate axis");
}

}  // namespace Test
}  // namespace Lotus

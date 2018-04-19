#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(SqueezeOpTest, Squeeze_1) {
  OpTester test("Squeeze");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddInput<float>("data", {1, 3, 4, 5}, std::vector<float>(60, 1.0f));
  test.AddOutput<float>("squeezed", {3, 4, 5}, std::vector<float>(60, 1.0f));
  test.Run();
}

TEST(SqueezeOpTest, Squeeze_2) {
  OpTester test("Squeeze");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2, 3});
  test.AddInput<float>("data", {1, 4, 1, 1, 2},
                       std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.AddOutput<float>("squeezed", {4, 2},
                        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.Run();
}

TEST(SqueezeOpTest, UnsortedAxes) {
  OpTester test("Squeeze");
  test.AddAttribute("axes", std::vector<int64_t>{3, 0, 2});
  test.AddInput<float>("data", {1, 4, 1, 1, 2},
                       std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.AddOutput<float>("squeezed", {4, 2},
                        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.Run();
}

TEST(SqueezeOpTest, DuplicateAxes) {
  OpTester test("Squeeze");
  test.AddAttribute("axes", std::vector<int64_t>{3, 0, 2, 0, 2, 3});
  test.AddInput<float>("data", {1, 4, 1, 1, 2},
                       std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.AddOutput<float>("squeezed", {4, 2},
                        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.Run();
}

TEST(SqueezeOpTest, BadAxes) {
  OpTester test("Squeeze");
  // Bad axes - should be 1 instead of 0.
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddInput<float>("data", {3, 1, 4, 5}, std::vector<float>(60, 1.0f));
  test.AddOutput<float>("squeezed", {3, 4, 5}, std::vector<float>(60, 1.0f));

  // Expect failure.
  test.Run(true);
}
}  // namespace Test
}  // namespace Lotus

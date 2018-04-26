#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(MathOpTest, FC2DTest) {
  OpTester test("FC");
  test.AddAttribute("axis", static_cast<int64_t>(1));
  test.AddAttribute("axis_w", static_cast<int64_t>(1));

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {3, 4}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
  test.Run();
}

TEST(MathOpTest, FCNDTest) {
  OpTester test("FC");
  test.AddAttribute("axis", static_cast<int64_t>(2));
  test.AddAttribute("axis_w", static_cast<int64_t>(1));

  test.AddInput<float>("A", {1, 2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {3, 2, 2}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
  test.Run();
}

TEST(MathOpTest, FCNDDefaultAxisTest) {
  OpTester test("FC");

  test.AddInput<float>("A", {2, 2, 2},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {3, 2, 2}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
  test.Run();
}

}  // namespace Test
}  // namespace Lotus

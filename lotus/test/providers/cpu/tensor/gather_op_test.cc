#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(GatherOpTest, Gather_axis0) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0L);
  test.AddInput<float>("data", {2L, 3L, 4L},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {1L}, {1L});
  test.AddOutput<float>("output", {1L, 3L, 4L},
                        {10.0f, 10.1f, 10.2f, 10.3f,
                         11.0f, 11.1f, 11.2f, 11.3f,
                         12.0f, 12.1f, 12.2f, 12.3f});
  test.Run();
}

TEST(GatherOpTest, Gather_negative_axis) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", -3L);
  test.AddInput<float>("data", {2L, 3L, 4L},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {1L}, {1L});
  test.AddOutput<float>("output", {1L, 3L, 4L},
                        {10.0f, 10.1f, 10.2f, 10.3f,
                         11.0f, 11.1f, 11.2f, 11.3f,
                         12.0f, 12.1f, 12.2f, 12.3f});
  test.Run();
}

TEST(GatherOpTest, Gather_invalid_axis) {
  OpTester test("Gather");
  // Invalid axis not in range [-r, r-1]
  test.AddAttribute<int64_t>("axis", -10L);
  test.AddInput<float>("data", {2L, 3L, 4L},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {1L}, {1L});
  test.AddOutput<float>("output", {1L, 3L, 4L},
                        {10.0f, 10.1f, 10.2f, 10.3f,
                         11.0f, 11.1f, 11.2f, 11.3f,
                         12.0f, 12.1f, 12.2f, 12.3f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "axis -10 is not in valid range [-3,2]");
}

TEST(GatherOpTest, Gather_axis1) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1L);
  test.AddInput<float>("data", {2L, 3L, 4L},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {2L}, {2L, 0L});
  test.AddOutput<float>("output", {2L, 2L, 4L},
                        {2.0f, 2.1f, 2.2f, 2.3f,
                         0.0f, 0.1f, 0.2f, 0.3f,
                         12.0f, 12.1f, 12.2f, 12.3f,
                         10.0f, 10.1f, 10.2f, 10.3f});
  test.Run();
}

TEST(GatherOpTest, Gather_axis2) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 2L);
  test.AddInput<float>("data", {2L, 3L, 4L},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {3L}, {1L, 0L, 2L});
  test.AddOutput<float>("output", {2L, 3L, 3L},
                        {0.1f, 0.0f, 0.2f,
                         1.1f, 1.0f, 1.2f,
                         2.1f, 2.0f, 2.2f,
                         10.1f, 10.0f, 10.2f,
                         11.1f, 11.0f, 11.2f,
                         12.1f, 12.0f, 12.2f});
  test.Run();
}

TEST(GatherOpTest, Gather_axis0_indices2d) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0L);
  test.AddInput<float>("data", {3L, 3L},
                       {0.0f, 0.1f, 0.2f,
                        1.0f, 1.1f, 1.2f,
                        2.0f, 2.1f, 2.2f});
  test.AddInput<int64_t>("indices", {2L, 2L},
                         {1L, 0L,
                          2L, 1L});
  test.AddOutput<float>("output", {2L, 2L, 3L},
                        {1.0f, 1.1f, 1.2f, 0.0f, 0.1f, 0.2f,
                         2.0f, 2.1f, 2.2f, 1.0f, 1.1f, 1.2f});
  test.Run();
}

TEST(GatherOpTest, Gather_axis1_indices2d) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1L);
  test.AddInput<float>("data", {3L, 3L},
                       {0.0f, 0.1f, 0.2f,
                        1.0f, 1.1f, 1.2f,
                        2.0f, 2.1f, 2.2f});
  test.AddInput<int64_t>("indices", {2L, 2L},
                         {1L, 0L,
                          2L, 1L});
  test.AddOutput<float>("output", {3L, 2L, 2L},
                        {0.1f, 0.0f, 0.2f, 0.1f,
                         1.1f, 1.0f, 1.2f, 1.1f,
                         2.1f, 2.0f, 2.2f, 2.1f});
  test.Run();
}
}  // namespace Test
}  // namespace Lotus

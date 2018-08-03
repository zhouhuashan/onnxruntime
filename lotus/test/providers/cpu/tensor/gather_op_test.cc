#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(GatherOpTest, Gather_axis0) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<float>("data", {2, 3, 4},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {1}, {1LL});
  test.AddOutput<float>("output", {1, 3, 4},
                        {10.0f, 10.1f, 10.2f, 10.3f,
                         11.0f, 11.1f, 11.2f, 11.3f,
                         12.0f, 12.1f, 12.2f, 12.3f});
  test.Run();
}

TEST(GatherOpTest, Gather_negative_axis) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", -3LL);
  test.AddInput<float>("data", {2, 3, 4},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {1}, {1LL});
  test.AddOutput<float>("output", {1, 3, 4},
                        {10.0f, 10.1f, 10.2f, 10.3f,
                         11.0f, 11.1f, 11.2f, 11.3f,
                         12.0f, 12.1f, 12.2f, 12.3f});
  test.Run();
}

TEST(GatherOpTest, Gather_invalid_axis) {
  OpTester test("Gather");
  // Invalid axis not in range [-r, r-1]
  test.AddAttribute<int64_t>("axis", -10LL);
  test.AddInput<float>("data", {2, 3, 4},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {1}, {1LL});
  test.AddOutput<float>("output", {1, 3, 4},
                        {10.0f, 10.1f, 10.2f, 10.3f,
                         11.0f, 11.1f, 11.2f, 11.3f,
                         12.0f, 12.1f, 12.2f, 12.3f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "axis -10 is not in valid range [-3,2]");
}

TEST(GatherOpTest, Gather_invalid_index) {
  OpTester test("Gather");
  // Invalid index 3. data[3] does not exist.
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<float>("data", {3, 4},
                       {0.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 7.0f,
                        8.0f, 9.0f, 10.0f, 11.0f});
  test.AddInput<int64_t>("indices", {3}, {0LL, 1LL, 3LL});
  test.AddOutput<float>("output", {1}, {1.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "indices element out of data bounds, idx=3 data_dim=3");
}

TEST(GatherOpTest, Gather_axis1) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<float>("data", {2, 3, 4},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {2}, {2LL, 0LL});
  test.AddOutput<float>("output", {2, 2, 4},
                        {2.0f, 2.1f, 2.2f, 2.3f,
                         0.0f, 0.1f, 0.2f, 0.3f,
                         12.0f, 12.1f, 12.2f, 12.3f,
                         10.0f, 10.1f, 10.2f, 10.3f});
  test.Run();
}

TEST(GatherOpTest, Gather_axis2) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 2LL);
  test.AddInput<float>("data", {2, 3, 4},
                       {0.0f, 0.1f, 0.2f, 0.3f,
                        1.0f, 1.1f, 1.2f, 1.3f,
                        2.0f, 2.1f, 2.2f, 2.3f,
                        10.0f, 10.1f, 10.2f, 10.3f,
                        11.0f, 11.1f, 11.2f, 11.3f,
                        12.0f, 12.1f, 12.2f, 12.3f});
  test.AddInput<int64_t>("indices", {3}, {1LL, 0LL, 2LL});
  test.AddOutput<float>("output", {2, 3, 3},
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
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<float>("data", {3, 3},
                       {0.0f, 0.1f, 0.2f,
                        1.0f, 1.1f, 1.2f,
                        2.0f, 2.1f, 2.2f});
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {1LL, 0LL,
                          2LL, 1LL});
  test.AddOutput<float>("output", {2, 2, 3},
                        {1.0f, 1.1f, 1.2f, 0.0f, 0.1f, 0.2f,
                         2.0f, 2.1f, 2.2f, 1.0f, 1.1f, 1.2f});
  test.Run();
}

TEST(GatherOpTest, Gather_axis1_indices2d) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<float>("data", {3, 3},
                       {0.0f, 0.1f, 0.2f,
                        1.0f, 1.1f, 1.2f,
                        2.0f, 2.1f, 2.2f});
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {1LL, 0LL,
                          2LL, 1LL});
  test.AddOutput<float>("output", {3, 2, 2},
                        {0.1f, 0.0f, 0.2f, 0.1f,
                         1.1f, 1.0f, 1.2f, 1.1f,
                         2.1f, 2.0f, 2.2f, 2.1f});
  test.Run();
}

TEST(GatherOpTest, Gather_axis1_indices2d_int32) {
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<int32_t>("data", {3, 3},
                          {0, 1, 2,
                           10, 11, 12,
                           20, 21, 22});
  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 0,
                          2, 1});
  test.AddOutput<int32_t>("output", {3, 2, 2},
                           {1, 0, 2, 1,
                            11, 10, 12, 11,
                            21, 20, 22, 21});
  test.Run();
}
}  // namespace Test
}  // namespace Lotus

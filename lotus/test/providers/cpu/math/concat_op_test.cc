#include "core/providers/cpu/misc/concat.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(MathOpTest, Concat1D) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<float>("input1", {1}, {1.0f});
  test.AddInput<float>("input2", {2}, {2.0f, 3.0f});
  test.AddInput<float>("input3", {4}, {4.0f, 5.0f, 6.0f, 7.0f});
  test.AddOutput<float>("concat_result", {7}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
  test.Run();
}

TEST(MathOpTest, Concat2D_1) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  std::vector<int64_t> dims{1, 4};
  test.AddInput<float>("input1", dims, {11.0f, 12.0f, 13.0f, 14.0f});
  test.AddInput<float>("input2", dims, {21.0f, 22.0f, 23.0f, 24.0f});
  test.AddInput<float>("input3", dims, {31.0f, 32.0f, 33.0f, 34.0f});
  test.AddOutput<float>("concat_result", {3, 4},
                        {11.0f, 12.0f, 13.0f, 14.0f,
                         21.0f, 22.0f, 23.0f, 24.0f,
                         31.0f, 32.0f, 33.0f, 34.0f});
  test.Run();
}

TEST(MathOpTest, Concat2D_2) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{4, 1};
  test.AddInput<float>("input1", dims, {11.0f, 21.0f, 31.0f, 41.0f});
  test.AddInput<float>("input2", {4, 2}, {12.0f, 13.0f, 22.0f, 23.0f, 32.0f, 33.0f, 42.0f, 43.0f});
  test.AddInput<float>("input3", dims, {14.0f, 24.0f, 34.0f, 44.0f});
  test.AddOutput<float>("concat_result", {4, 4},
                        {11.0f, 12.0f, 13.0f, 14.0f,
                         21.0f, 22.0f, 23.0f, 24.0f,
                         31.0f, 32.0f, 33.0f, 34.0f,
                         41.0f, 42.0f, 43.0f, 44.0f});
  test.Run();
}

TEST(MathOpTest, Concat3D_1) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  std::vector<int64_t> dims{1, 3, 3};
  test.AddInput<float>("input1", dims, {111.0f, 112.0f, 113.0f, 121.0f, 122.0f, 123.0f, 131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", dims, {211.0f, 212.0f, 213.0f, 221.0f, 222.0f, 223.0f, 231.0f, 232.0f, 233.0f});
  test.AddInput<float>("input3", dims, {311.0f, 312.0f, 313.0f, 321.0f, 322.0f, 323.0f, 331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,
                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,
                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(MathOpTest, Concat3D_2) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{3, 1, 3};
  test.AddInput<float>("input1", dims, {111.0f, 112.0f, 113.0f, 211.0f, 212.0f, 213.0f, 311.0f, 312.0f, 313.0f});
  test.AddInput<float>("input2", dims, {121.0f, 122.0f, 123.0f, 221.0f, 222.0f, 223.0f, 321.0f, 322.0f, 323.0f});
  test.AddInput<float>("input3", dims, {131.0f, 132.0f, 133.0f, 231.0f, 232.0f, 233.0f, 331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,
                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,
                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

}  // namespace Test
}  // namespace Lotus

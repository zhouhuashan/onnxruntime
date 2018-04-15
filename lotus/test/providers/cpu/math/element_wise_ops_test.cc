#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(MathOpTest, Add) {
  OpTester test("Add");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f,
                        -5.4f, 9.3f, -10'000.0f});
  test.AddInput<float>("B", dims,
                       {-1.0f, 4.4f, 432.3f,
                        0.0f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 10'000.0f});
  test.AddOutput<float>("C", dims,
                        {0.0f, 6.4f, 431.3f,
                         0.0f, 5.0f, -36.0f,
                         -10.8f, 18.6f, 0.0f});
  test.Run();
}

TEST(MathOpTest, Add_Broadcast_Axis) {
  OpTester test("Add");

  test.AddAttribute("axis", int64_t{0});
  test.AddAttribute("broadcast", int64_t{1});

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f});
  test.AddInput<float>("B", {3},
                       {3.0f,
                        2.0f,
                        1.0f});
  test.AddOutput<float>("C", dims,
                        {4.0f, 5.0f, 6.0f,
                         6.0f, 7.0f, 8.0f,
                         8.0f, 9.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Add_Broadcast) {
  OpTester test("Add");

  test.AddAttribute("broadcast", int64_t{1});

  std::vector<int64_t> dims{3, 2};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f});
  test.AddInput<float>("B", {3},
                       {1.0f,
                        2.0f,
                        3.0f});
  test.AddOutput<float>("C", dims,
                        {2.0f, 3.0f,
                         5.0f, 6.0f,
                         8.0f, 9.0f});
  test.Run();
}

TEST(MathOpTest, Sub) {
  OpTester test("Sub");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f,
                        -5.4f, 9.3f, -10'000.0f});
  test.AddInput<float>("B", dims,
                       {-1.0f, 4.4f, 432.3f,
                        0.0f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 10'000.0f});
  test.AddOutput<float>("C", dims,
                        {2.0f, -2.4f, -433.3f,
                         0.0f, -2.0f, -164.0f,
                         0.0f, 0.0f, -20'000.0f});
  test.Run();
}

TEST(MathOpTest, Mul) {
  OpTester test("Mul");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f, -5.4f,
                        9.30f, -10'000.0f});
  test.AddInput<float>("B", dims,
                       {-1.0f, 4.4f, 432.3f,
                        0.0f, 3.5f, 64.0f, -5.4f,
                        9.30f, 10'000.0f});
  test.AddOutput<float>("C", dims,
                        {-1.0f, 8.8f, -432.3f,
                         0.0f, 5.25f, -6'400.0f,
                         29.16f, 86.49f, -100'000'000.0f});
  test.Run();
}

TEST(MathOpTest, Div) {
  OpTester test("Div");
  std::vector<int64_t> dims{2, 3};
  test.AddInput<float>("A", dims,
                       {1'000.0f, 1.0f, 6.0f,
                        0.0f, -10.0f, -1.0f});
  test.AddInput<float>("B", dims,
                       {1'000.0f, 2.0f, 3.0f,
                        1.0f, -1.0f, 4.0f});
  test.AddOutput<float>("C", dims,
                        {1.0f, 0.5f, 2.0f,
                         0.0f, 10.0f, -0.25f});
  test.Run();
}

TEST(MathOpTest, Abs) {
  OpTester test("Abs");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, -2.0f, -0.0f, -10.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 0.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Neg) {
  OpTester test("Neg");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, -2.0f,
                        0.0f, -10.0f});
  test.AddOutput<float>("Y", dims,
                        {-1.0f, 2.0f,
                         -0.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Floor) {
  OpTester test("Floor");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {-1.5f, 0.2f,
                        -0.5f, 10.3f});
  test.AddOutput<float>("Y", dims,
                        {-2.0f, 0.0f,
                         -1.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Ceil) {
  OpTester test("Ceil");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {-1.5f, 0.2f,
                        -0.5f, 10.3f});
  test.AddOutput<float>("Y", dims,
                        {-1.0f, 1.0f,
                         0.0f, 11.0f});
  test.Run();
}

TEST(MathOpTest, Reciprocal) {
  OpTester test("Reciprocal");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, 2.0f,
                        -1.0f, -2.0f});
  test.AddOutput<float>("Y", dims,
                        {1.0f, 0.5f,
                         -1.0f, -0.5f});
  test.Run();
}

TEST(MathOpTest, Sqrt) {
  OpTester test("Sqrt");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, 4.0f,
                        0.0f, 9.0f});
  test.AddOutput<float>("Y", dims,
                        {1.0f, 2.0f,
                         0.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Pow) {
  OpTester test("Pow");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {2.0f, 2.0f,
                        sqrt(2.0f), 1.0f});
  test.AddInput<float>("Y", dims,
                       {0.0f, 8.0f,
                        2.0f, 9.0f});
  test.AddOutput<float>("Z", dims,
                        {1.0f, 256.0f,
                         2.0f, 1.0f});
  test.Run();
}

TEST(MathOpTest, Pow_Broadcast_Scalar) {
  OpTester test("Pow");

  test.AddAttribute("broadcast", int64_t{1});

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("Y", {}, {2.0f});
  test.AddOutput<float>("Z", dims, {1.0f, 4.0f, 9.0f});
  test.Run();
}

TEST(MathOpTest, Exp) {
  OpTester test("Exp");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {0.0f, 1.0f,
                        2.0f, 10.0f});
  test.AddOutput<float>("Y", dims,
                        {1.0f, exp(1.0f),
                         exp(2.0f), exp(10.0f)});
  test.Run();
}

TEST(MathOpTest, Log) {
  OpTester test("Log");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, 2.0f,
                        5.0f, 10.0f});
  test.AddOutput<float>("Y", dims,
                        {0.0f, log(2.0f),
                         log(5.0f), log(10.0f)});
  test.Run();
}

TEST(MathOpTest, Sum) {
  OpTester test("Sum");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.25f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("sum", dims,
                        {3.0f, 0.0f, 6.0f,
                         -6.0f, 6.6f, 28.0f,
                         -1.0f, 0.06f, 0.25f});
  test.Run();
}

TEST(MathOpTest, Min) {
  OpTester test("Min");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("sum", dims,
                        {1.0f, 0.0f, 1.0f,
                         -3.0f, 1.1f, -100.0f,
                         -5.4f, 0.01f, -10'000.0f});
  test.Run();
}

TEST(MathOpTest, Max) {
  OpTester test("Max");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_2", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("sum", dims,
                        {1.0f, 0.0f, 3.0f,
                         -1.0f, 3.3f, 64.0f,
                         5.4f, 0.03f, 10'000.0f});
  test.Run();
}

TEST(MathOpTest, And) {
  OpTester test("And");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, false, false, true});
  test.Run();
}

TEST(MathOpTest, Or) {
  OpTester test("Or");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run();
}

TEST(MathOpTest, Xor) {
  OpTester test("Xor");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Xor_bcast3v2d) {
  OpTester test("Xor");
  test.AddAttribute("broadcast", int64_t{1});

  test.AddInput<bool>("A", {2, 3, 4},
                      {false, true, false, true,
                       false, true, false, true,
                       false, true, false, true,

                       false, true, false, true,
                       false, true, false, true,
                       false, true, false, true});
  test.AddInput<bool>("B", {3, 4},
                      {false, false, true, true,
                       false, false, true, true,
                       false, false, true, true});
  test.AddOutput<bool>("C", {2, 3, 4},
                       {false, true, true, false,
                        false, true, true, false,
                        false, true, true, false,

                        false, true, true, false,
                        false, true, true, false,
                        false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Less) {
  OpTester test("Less");
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Greater) {
  OpTester test("Greater");
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {false, false, false, false});
  test.Run();
}

TEST(MathOpTest, Equal) {
  OpTester test("Equal");
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int32_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Mean) {
  OpTester test("Mean");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.0f, 0.01f, -10.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 65.0f,
                        -1.0f, 0.02f, -1.0f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 65.0f,
                        -3.0f, 0.03f, -1.0f});
  test.AddOutput<float>("mean", dims,
                        {1.0f, 0.0f, 2.0f,
                         -2.0f, 2.2f, 10.0f,
                         -3.0f, 0.02f, -4.0f});
  test.Run();
}
}  // namespace Test
}  // namespace Lotus

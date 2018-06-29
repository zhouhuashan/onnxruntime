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
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Add_Broadcast_Axis) {
  OpTester test("Add");

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f});
  test.AddInput<float>("B", {3, 1},
                       {3.0f,
                        2.0f,
                        1.0f});
  test.AddOutput<float>("C", dims,
                        {4.0f, 5.0f, 6.0f,
                         6.0f, 7.0f, 8.0f,
                         8.0f, 9.0f, 10.0f});
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Add_Broadcast_0x0) {
  OpTester test("Add");

  test.AddInput<float>("A", {}, {10.0f});
  test.AddInput<float>("B", {}, {2.0f});
  test.AddOutput<float>("C", {}, {12.0f});
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Add_Broadcast_0x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {}, {10.0f});
  test.AddInput<float>("B", {1}, {2.0f});
  test.AddOutput<float>("C", {1}, {12.0f});
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Add_Broadcast_1x0) {
  OpTester test("Add");

  test.AddInput<float>("A", {1}, {10.0f});
  test.AddInput<float>("B", {}, {2.0f});
  test.AddOutput<float>("C", {1}, {12.0f});
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Add_Broadcast_1x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {1}, {10.0f});
  test.AddInput<float>("B", {1}, {2.0f});
  test.AddOutput<float>("C", {1}, {12.0f});
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Add_Broadcast_3x2_3x1) {
  OpTester test("Add");

  std::vector<int64_t> dims{3, 2};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f});
  test.AddInput<float>("B", {3, 1},
                       {1.0f,
                        2.0f,
                        3.0f});
  test.AddOutput<float>("C", dims,
                        {2.0f, 3.0f,
                         5.0f, 6.0f,
                         8.0f, 9.0f});
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Add_Broadcast_2x1x4_1x3x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {2, 1, 4},
                       {101.0f, 102.0f, 103.0f, 104.0f,
                        201.0f, 202.0f, 203.0f, 204.0f});
  test.AddInput<float>("B", {1, 3, 1},
                       {010.0f, 020.0f, 030.0f});
  test.AddOutput<float>("C", {2, 3, 4},
                        {111.0f, 112.0f, 113.0f, 114.0f,
                         121.0f, 122.0f, 123.0f, 124.0f,
                         131.0f, 132.0f, 133.0f, 134.0f,

                         211.0f, 212.0f, 213.0f, 214.0f,
                         221.0f, 222.0f, 223.0f, 224.0f,
                         231.0f, 232.0f, 233.0f, 234.0f});
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Add_Broadcast_2x1x1_3x4) {
  OpTester test("Add");

  test.AddInput<float>("A", {2, 1, 1},
                       {100.0f, 200.0f});
  test.AddInput<float>("B", {3, 4},
                       {011.0f, 012.0f, 013.0f, 014.0f,
                        021.0f, 022.0f, 023.0f, 024.0f,
                        031.0f, 032.0f, 033.0f, 034.0f});
  test.AddOutput<float>("C", {2, 3, 4},
                        {111.0f, 112.0f, 113.0f, 114.0f,
                         121.0f, 122.0f, 123.0f, 124.0f,
                         131.0f, 132.0f, 133.0f, 134.0f,

                         211.0f, 212.0f, 213.0f, 214.0f,
                         221.0f, 222.0f, 223.0f, 224.0f,
                         231.0f, 232.0f, 233.0f, 234.0f});
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Sub_Broadcast_Scalar) {
  OpTester test("Sub");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f,
                        -5.4f, 9.3f, -10'000.0f});
  test.AddInput<float>("B", {}, {5.0f});
  test.AddOutput<float>("C", dims,
                        {-4.0f, -3.0f, -6.0f,
                         -5.0f, -3.5f, -105.0f,
                         -10.4f, 4.3f, -10'005.0f});
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Abs) {
  OpTester test("Abs");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, -2.0f, -0.0f, -10.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 0.0f, 10.0f});
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Pow_Broadcast_Scalar) {
  OpTester test("Pow");

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("Y", {}, {2.0f});
  test.AddOutput<float>("Z", dims, {1.0f, 4.0f, 9.0f});
  test.RunOnCpuAndCuda();
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
  test.SetOutputRelErr("Y", 1e-7f);
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
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
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Or) {
  OpTester test("Or");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Xor) {
  OpTester test("Xor");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, true, true, false});
  test.RunOnCpuAndCuda();
}

TEST(MathOpTest, Xor_bcast3v2d) {
  OpTester test("Xor");

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
  test.RunOnCpuAndCuda();
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

TEST(MathOpTest, AffineDefaultAttributes) {
  OpTester test("Affine");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Affine) {
  OpTester test("Affine");
  std::vector<int64_t> dims{2, 2};
  test.AddAttribute("alpha", 2.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {1.0f, 3.0f, 5.0f, 7.0f});
  test.Run();
}

template <float (&op)(float value)>
void TrigTest(OpTester &test, std::initializer_list<float> input) {
  std::vector<int64_t> dims{static_cast<int64_t>(input.size())};

  std::vector<float> output;
  for (auto v : input)
    output.push_back(op(v));

  test.AddInput<float>("X", dims, input);
  test.AddOutput<float>("Y", dims, output);
  test.Run();
}

TEST(MathOpTest, Sin) {
  OpTester test("Sin");
  TrigTest<std::sin>(test, {1.1f, -1.1f, 2.2f, -2.2f});
}

TEST(MathOpTest, Cos) {
  OpTester test("Cos");
  TrigTest<std::cos>(test, {1.1f, -1.1f, 2.2f, -2.2f});
}

TEST(MathOpTest, Tan) {
  OpTester test("Tan");
  TrigTest<std::tan>(test, {-100.0f, -50.0f, 0.0f, 50.0f, 100.0f});
}

TEST(MathOpTest, Asin) {
  OpTester test("Asin");
  TrigTest<std::asin>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Acos) {
  OpTester test("Acos");
  TrigTest<std::acos>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Atan) {
  OpTester test("Atan");
  TrigTest<std::atan>(test, {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f});
}

}  // namespace Test
}  // namespace Lotus

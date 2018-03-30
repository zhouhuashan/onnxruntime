#include "core/providers/cpu/math/element_wise_ops.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

static const TypeProto_Set s_typeProto_float{TensorProto_DataType_FLOAT};

TEST(MathOpTest, Add) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float),
    input2_def("B", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model("Add", {&input1_def, &input2_def}, {&output_def});
  SimpleFloatTest<Add> test(model);

  std::vector<int64_t> dims{3, 3};
  test.AddInput(dims, {1.0f, 2.0f, -1.0f, 0.0f, 1.5f, -100.0f, -5.4f, 9.3f, -10'000.0f});
  test.AddInput(dims, {-1.0f, 4.4f, 432.3f, 0.0f, 3.5f, 64.0f, -5.4f, 9.3f, 10'000.0f});
  test.AddOutput(dims);
  float expected_vals[]{0.0f, 6.4f, 431.3f, 0.0f, 5.0f, -36.0f, -10.8f, 18.6f, 0.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Add_Broadcast_Axis) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float),
    input2_def("B", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model("Add", {&input1_def, &input2_def}, {&output_def});

  EXPECT_TRUE(model.Node().AddAttribute("axis", int64_t{0}));
  EXPECT_TRUE(model.Node().AddAttribute("broadcast", int64_t{1}));

  SimpleFloatTest<Add> test(model);

  std::vector<int64_t> dims{3, 3};
  test.AddInput(dims, {1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f,
                       7.0f, 8.0f, 9.0f});
  test.AddInput({3}, {3.0f,
                      2.0f,
                      1.0f});
  test.AddOutput(dims);
  float expected_vals[]{4.0f, 5.0f, 6.0f,
                        6.0f, 7.0f, 8.0f,
                        8.0f, 9.0f, 10.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Add_Broadcast) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float),
    input2_def("B", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model("Add", {&input1_def, &input2_def}, {&output_def});

  EXPECT_TRUE(model.Node().AddAttribute("broadcast", int64_t{1}));

  SimpleFloatTest<Add> test(model);

  std::vector<int64_t> dims{3, 2};
  test.AddInput(dims, {1.0f, 2.0f,
                       3.0f, 4.0f,
                       5.0f, 6.0f});
  test.AddInput({3}, {1.0f,
                      2.0f,
                      3.0f});
  test.AddOutput(dims);
  float expected_vals[]{2.0f, 3.0f,
                        5.0f, 6.0f,
                        8.0f, 9.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Sub) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float),
    input2_def("B", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model("Sub", {&input1_def, &input2_def}, {&output_def});
  SimpleFloatTest<Sub> test(model);

  std::vector<int64_t> dims{3, 3};
  test.AddInput(dims, {1.0f, 2.0f, -1.0f, 0.0f, 1.5f, -100.0f, -5.4f, 9.3f, -10'000.0f});
  test.AddInput(dims, {-1.0f, 4.4f, 432.3f, 0.0f, 3.5f, 64.0f, -5.4f, 9.3f, 10'000.0f});
  test.AddOutput(dims);
  float expected_vals[]{2.0f, -2.4f, -433.3f, 0.0f, -2.0f, -164.0f, 0.0f, 0.0f, -20'000.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Mul) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float), input2_def("B", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model("Mul", {&input1_def, &input2_def}, {&output_def});
  SimpleFloatTest<Mul> test(model);

  std::vector<int64_t> dims{3, 3};
  test.AddInput(dims, {1.0f, 2.0f, -1.0f, 0.0f, 1.5f, -100.0f, -5.4f, 9.30f, -10'000.0f});
  test.AddInput(dims, {-1.0f, 4.4f, 432.3f, 0.0f, 3.5f, 64.0f, -5.4f, 9.30f, 10'000.0f});
  test.AddOutput(dims);
  float expected_vals[]{-1.0f, 8.8f, -432.3f, 0.0f, 5.25f, -6'400.0f, 29.16f, 86.49f, -100'000'000.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Div) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float),
    input2_def("B", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model("Div", {&input1_def, &input2_def}, {&output_def});
  SimpleFloatTest<Div> test(model);

  std::vector<int64_t> dims{2, 3};
  test.AddInput(dims, {1'000.0f, 1.0f, 6.0f,
                       0.0f, -10.0f, -1.0f});
  test.AddInput(dims, {1'000.0f, 2.0f, 3.0f,
                       1.0f, -1.0f, 4.0f});
  test.AddOutput(dims);
  float expected_vals[]{1.0f, 0.5f, 2.0f,
                        0.0f, 10.0f, -0.25f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Abs) {
  LotusIR::NodeArg input_def("X", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  TestModel model("Abs", {&input_def}, {&output_def});
  SimpleFloatTest<Abs> test(model);

  std::vector<int64_t> dims{2, 2};
  test.AddInput(dims, {1.0f, -2.0f, -0.0f, -10.0f});
  test.AddOutput(dims);
  float expected_vals[]{1.0f, 2.0f, 0.0f, 10.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Neg) {
  LotusIR::NodeArg input_def("X", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  TestModel model("Neg", {&input_def}, {&output_def});
  SimpleFloatTest<Neg> test(model);

  std::vector<int64_t> dims{2, 2};
  test.AddInput(dims, {1.0f, -2.0f, 0.0f, -10.0f});
  test.AddOutput(dims);
  float expected_vals[]{-1.0f, 2.0f, -0.0f, 10.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Floor) {
  LotusIR::NodeArg input_def("X", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  TestModel model("Floor", {&input_def}, {&output_def});
  SimpleFloatTest<Floor> test(model);

  std::vector<int64_t> dims{2, 2};
  test.AddInput(dims, {-1.5f, 0.2f, -0.5f, 10.3f});
  test.AddOutput(dims);
  float expected_vals[]{-2.0f, 0.0f, -1.0f, 10.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Ceil) {
  LotusIR::NodeArg input_def("X", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  TestModel model("Ceil", {&input_def}, {&output_def});
  SimpleFloatTest<Ceil> test(model);

  std::vector<int64_t> dims{2, 2};
  test.AddInput(dims, {-1.5f, 0.2f, -0.5f, 10.3f});
  test.AddOutput(dims);
  float expected_vals[]{-1.0f, 1.0f, 0.0f, 11.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Reciprocal) {
  LotusIR::NodeArg input_def("X", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  TestModel model("Reciprocal", {&input_def}, {&output_def});
  SimpleFloatTest<Reciprocal> test(model);

  std::vector<int64_t> dims{2, 2};
  test.AddInput(dims, {1.0f, 2.0f, -1.0f, -2.0f});
  test.AddOutput(dims);
  float expected_vals[]{1.0f, 0.5f, -1.0f, -0.5f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Sqrt) {
  LotusIR::NodeArg input_def("X", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  TestModel model("Sqrt", {&input_def}, {&output_def});
  SimpleFloatTest<Sqrt> test(model);

  std::vector<int64_t> dims{2, 2};
  test.AddInput(dims, {1.0f, 4.0f, 0.0f, 9.0f});
  test.AddOutput(dims);
  float expected_vals[]{1.0f, 2.0f, 0.0f, 3.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Pow) {
  LotusIR::NodeArg input1_def("X", &s_typeProto_float),
    input2_def("Y", &s_typeProto_float), output_def("Z", &s_typeProto_float);
  TestModel model("Pow", {&input1_def, &input2_def}, {&output_def});
  SimpleFloatTest<Pow> test(model);

  std::vector<int64_t> dims{2, 2};
  test.AddInput(dims, {2.0f, 2.0f, sqrt(2.0f), 1.0f});
  test.AddInput(dims, {0.0f, 8.0f, 2.0f, 9.0f});
  test.AddOutput(dims);
  float expected_vals[]{1.0f, 256.0f, 2.0f, 1.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Pow_Broadcast_Scalar) {
  LotusIR::NodeArg input1_def("X", &s_typeProto_float),
    input2_def("Y", &s_typeProto_float), output_def("Z", &s_typeProto_float);
  TestModel model("Pow", {&input1_def, &input2_def}, {&output_def});

  EXPECT_TRUE(model.Node().AddAttribute("broadcast", int64_t{1}));

  SimpleFloatTest<Pow> test(model);

  std::vector<int64_t> dims{3};
  test.AddInput(dims, {1.0f, 2.0f, 3.0f});
  test.AddInput({}, {2.0f});
  test.AddOutput(dims);
  float expected_vals[]{1.0f, 4.0f, 9.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Exp) {
  LotusIR::NodeArg input_def("X", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  TestModel model("Exp", {&input_def}, {&output_def});
  SimpleFloatTest<Exp> test(model);

  std::vector<int64_t> dims{2, 2};
  test.AddInput(dims, {0.0f, 1.0f, 2.0f, 10.0f});
  test.AddOutput(dims);
  float expected_vals[]{1.0f, exp(1.0f), exp(2.0f), exp(10.0f)};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Log) {
  LotusIR::NodeArg input_def("X", &s_typeProto_float), output_def("Y", &s_typeProto_float);
  TestModel model("Log", {&input_def}, {&output_def});
  SimpleFloatTest<Log> test(model);

  std::vector<int64_t> dims{2, 2};
  test.AddInput(dims, {1.0f, 2.0f, 5.0f, 10.0f});
  test.AddOutput(dims);
  float expected_vals[]{0.0f, log(2.0f), log(5.0f), log(10.0f)};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Sum) {
  LotusIR::NodeArg input1_def("data_0", &s_typeProto_float),
    input2_def("data_1", &s_typeProto_float), input3_def("data_3", &s_typeProto_float),
    output_def("sum", &s_typeProto_float);
  TestModel model("Sum", {&input1_def, &input2_def, &input3_def}, {&output_def});
  SimpleFloatTest<Sum> test(model);

  std::vector<int64_t> dims{3, 3};
  test.AddInput(dims, {1.0f, 0.0f, 1.0f, -1.0f, 1.1f, -100.0f, -5.4f, 0.01f, -10'000.0f});
  test.AddInput(dims, {1.0f, 0.0f, 2.0f, -2.0f, 2.2f, 64.0f, -1.0f, 0.02f, 0.1f});
  test.AddInput(dims, {1.0f, 0.0f, 3.0f, -3.0f, 3.3f, 64.0f, 5.4f, 0.03f, 10'000.0f});
  test.AddOutput(dims);
  float expected_vals[]{3.0f, 0.0f, 6.0f, -6.0f, 6.6f, 28.0f, -1.0f, 0.06f, 0.1f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Min) {
  LotusIR::NodeArg input1_def("data_0", &s_typeProto_float), input2_def("data_1", &s_typeProto_float), input3_def("data_3", &s_typeProto_float), output_def("sum", &s_typeProto_float);
  TestModel model("Min", {&input1_def, &input2_def, &input3_def}, {&output_def});
  SimpleFloatTest<Min> test(model);

  std::vector<int64_t> dims{3, 3};
  test.AddInput(dims, {1.0f, 0.0f, 1.0f, -1.0f, 1.1f, -100.0f, -5.4f, 0.01f, -10'000.0f});
  test.AddInput(dims, {1.0f, 0.0f, 2.0f, -2.0f, 2.2f, 64.0f, -1.0f, 0.02f, 0.1f});
  test.AddInput(dims, {1.0f, 0.0f, 3.0f, -3.0f, 3.3f, 64.0f, 5.4f, 0.03f, 10'000.0f});
  test.AddOutput(dims);
  float expected_vals[]{1.0f, 0.0f, 1.0f, -3.0f, 1.1f, -100.0f, -5.4f, 0.01f, -10'000.0f};
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Max) {
  LotusIR::NodeArg input1_def("data_0", &s_typeProto_float), input2_def("data_1", &s_typeProto_float), input3_def("data_3", &s_typeProto_float), output_def("sum", &s_typeProto_float);
  TestModel model("Max", {&input1_def, &input2_def, &input3_def}, {&output_def});
  SimpleFloatTest<Max> test(model);

  std::vector<int64_t> dims{3, 3};
  test.AddInput(dims, {1.0f, 0.0f, 1.0f, -1.0f, 1.1f, -100.0f, -5.4f, 0.01f, -10'000.0f});
  test.AddInput(dims, {1.0f, 0.0f, 2.0f, -2.0f, 2.2f, 64.0f, -1.0f, 0.02f, 0.1f});
  test.AddInput(dims, {1.0f, 0.0f, 3.0f, -3.0f, 3.3f, 64.0f, 5.4f, 0.03f, 10'000.0f});
  test.AddOutput(dims);
  float expected_vals[]{1.0f, 0.0f, 3.0f, -1.0f, 3.3f, 64.0f, 5.4f, 0.03f, 10'000.0f};
  test.Run(dims, expected_vals);
}

}  // namespace Test
}  // namespace Lotus

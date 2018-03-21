#include "core/providers/cpu/misc/concat.h"
#include "core/providers/cpu/misc/constant.h"
#include "gtest/gtest.h"
#include "test/test_utils.h"

namespace Lotus {
namespace Test {

TypeProto_Set s_typeProto_float{TensorProto_DataType_FLOAT};

TEST(MathOpTest, Constant) {
  LotusIR::NodeArg output_def("output", &s_typeProto_float);
  TestModel model("Constant", {}, {&output_def});

  std::vector<int64_t> dims{2, 3};
  float expected_vals[] = {11.0f, 12.0f, 13.0f, 21.0f, 22.0f, 33.0f};

  TensorProto t;
  t.set_data_type(TensorProto_DataType_FLOAT);
  for (auto v : dims)
    *t.mutable_dims()->Add() = v;
  for (auto v : expected_vals)
    *t.mutable_float_data()->Add() = v;

  EXPECT_TRUE(model.Node().AddAttribute("value", t));

  SimpleFloatTest<Constant> test(model);
  test.Run(dims, expected_vals);
}

TEST(MathOpTest, Concat1D) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float), input2_def("B", &s_typeProto_float), input3_def("A3", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model{"Concat", {&input1_def, &input2_def, &input3_def}, {&output_def}};

  EXPECT_TRUE(model.Node().AddAttribute("axis", 0LL));

  SimpleFloatTest<Concat> test{model};

  test.AddInput({1}, {1.0f});
  test.AddInput({2}, {2.0f, 3.0f});
  test.AddInput({4}, {4.0f, 5.0f, 6.0f, 7.0f});

  std::vector<int64_t> expected_dims{7};
  test.AddOutput(expected_dims);
  float expected_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  test.Run(expected_dims, expected_vals);
}

TEST(MathOpTest, Concat2D_1) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float), input2_def("B", &s_typeProto_float), input3_def("A3", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model{"Concat", {&input1_def, &input2_def, &input3_def}, {&output_def}};

  EXPECT_TRUE(model.Node().AddAttribute("axis", 1LL));

  SimpleFloatTest<Concat> test{model};

  std::vector<int64_t> dims{4, 1};
  test.AddInput(dims, {11.0f, 12.0f, 13.0f, 14.0f});
  test.AddInput(dims, {21.0f, 22.0f, 23.0f, 24.0f});
  test.AddInput(dims, {31.0f, 32.0f, 33.0f, 34.0f});

  std::vector<int64_t> expected_dims{4, 3};
  test.AddOutput(expected_dims);
  float expected_vals[] = {11.0f, 12.0f, 13.0f, 14.0f,
                           21.0f, 22.0f, 23.0f, 24.0f,
                           31.0f, 32.0f, 33.0f, 34.0f};

  test.Run(expected_dims, expected_vals);
}

TEST(MathOpTest, Concat2D_2) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float), input2_def("B", &s_typeProto_float), input3_def("A3", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model{"Concat", {&input1_def, &input2_def, &input3_def}, {&output_def}};

  EXPECT_TRUE(model.Node().AddAttribute("axis", 0LL));

  SimpleFloatTest<Concat> test{model};

  std::vector<int64_t> dims{1, 4};
  test.AddInput(dims, {11.0f,
                       21.0f,
                       31.0f,
                       41.0f});
  test.AddInput({2, 4}, {12.0f, 13.0f,
                         22.0f, 23.0f,
                         32.0f, 33.0f,
                         42.0f, 43.0f});
  test.AddInput(dims, {14.0f,
                       24.0f,
                       34.0f,
                       44.0f});

  std::vector<int64_t> expected_dims{4, 4};
  test.AddOutput(expected_dims);
  float expected_vals[] = {11.0f, 12.0f, 13.0f, 14.0f,
                           21.0f, 22.0f, 23.0f, 24.0f,
                           31.0f, 32.0f, 33.0f, 34.0f,
                           41.0f, 42.0f, 43.0f, 44.0f};

  test.Run(expected_dims, expected_vals);
}

TEST(MathOpTest, Concat3D_1) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float), input2_def("B", &s_typeProto_float), input3_def("A3", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model{"Concat", {&input1_def, &input2_def, &input3_def}, {&output_def}};

  EXPECT_TRUE(model.Node().AddAttribute("axis", 2LL));
  SimpleFloatTest<Concat> test{model};

  std::vector<int64_t> dims{3, 3, 1};
  test.AddInput(dims, {111.0f, 112.0f, 113.0f,
                       121.0f, 122.0f, 123.0f,
                       131.0f, 132.0f, 133.0f});
  test.AddInput(dims, {211.0f, 212.0f, 213.0f,
                       221.0f, 222.0f, 223.0f,
                       231.0f, 232.0f, 233.0f});
  test.AddInput(dims, {311.0f, 312.0f, 313.0f,
                       321.0f, 322.0f, 323.0f,
                       331.0f, 332.0f, 333.0f});

  std::vector<int64_t> expected_dims{3, 3, 3};
  test.AddOutput(expected_dims);
  float expected_vals[] = {111.0f, 112.0f, 113.0f,
                           121.0f, 122.0f, 123.0f,
                           131.0f, 132.0f, 133.0f,
                           211.0f, 212.0f, 213.0f,
                           221.0f, 222.0f, 223.0f,
                           231.0f, 232.0f, 233.0f,
                           311.0f, 312.0f, 313.0f,
                           321.0f, 322.0f, 323.0f,
                           331.0f, 332.0f, 333.0f};

  test.Run(expected_dims, expected_vals);
}

TEST(MathOpTest, Concat3D_2) {
  LotusIR::NodeArg input1_def("A", &s_typeProto_float), input2_def("B", &s_typeProto_float), input3_def("A3", &s_typeProto_float), output_def("C", &s_typeProto_float);
  TestModel model{"Concat", {&input1_def, &input2_def, &input3_def}, {&output_def}};

  EXPECT_TRUE(model.Node().AddAttribute("axis", 1LL));
  SimpleFloatTest<Concat> test{model};

  std::vector<int64_t> dims{3, 1, 3};
  test.AddInput(dims, {111.0f, 112.0f, 113.0f,
                       211.0f, 212.0f, 213.0f,
                       311.0f, 312.0f, 313.0f});
  test.AddInput(dims, {121.0f, 122.0f, 123.0f,
                       221.0f, 222.0f, 223.0f,
                       321.0f, 322.0f, 323.0f});
  test.AddInput(dims, {131.0f, 132.0f, 133.0f,
                       231.0f, 232.0f, 233.0f,
                       331.0f, 332.0f, 333.0f});

  std::vector<int64_t> expected_dims{3, 3, 3};
  test.AddOutput(expected_dims);
  float expected_vals[] = {111.0f, 112.0f, 113.0f,
                           121.0f, 122.0f, 123.0f,
                           131.0f, 132.0f, 133.0f,
                           211.0f, 212.0f, 213.0f,
                           221.0f, 222.0f, 223.0f,
                           231.0f, 232.0f, 233.0f,
                           311.0f, 312.0f, 313.0f,
                           321.0f, 322.0f, 323.0f,
                           331.0f, 332.0f, 333.0f};

  test.Run(expected_dims, expected_vals);
}

}  // namespace Test
}  // namespace Lotus

#include "core/providers/cpu/misc/constant.h"
#include "gtest/gtest.h"
#include "test/test_utils.h"

namespace Lotus {
namespace Test {

static const TypeProto_Set s_typeProto_float{TensorProto_DataType_FLOAT};

TEST(MathOpTest, Constant) {
  LotusIR::NodeArg output_def("output", &s_typeProto_float);
  TestModel model("Constant", {}, {&output_def});

  std::vector<int64_t> dims{2, 3};
  float expected_vals[]{11.0f, 12.0f, 13.0f, 21.0f, 22.0f, 33.0f};

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

}  // namespace Test
}  // namespace Lotus

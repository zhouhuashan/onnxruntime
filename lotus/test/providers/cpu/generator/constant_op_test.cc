#include "core/providers/cpu/generator/constant.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(MathOpTest, Constant) {
  OpTester test("Constant");

  std::vector<int64_t> dims{2, 3};
  auto expected_vals = {11.0f, 12.0f, 13.0f,
                        21.0f, 22.0f, 33.0f};

  TensorProto t;
  t.set_data_type(TensorProto_DataType_FLOAT);

  for (auto v : dims)
    *t.mutable_dims()->Add() = v;

  for (auto v : expected_vals)
    *t.mutable_float_data()->Add() = v;

  test.AddAttribute("value", t);

  test.AddOutput<float>("output", dims, expected_vals);
  test.Run();
}

}  // namespace Test
}  // namespace Lotus

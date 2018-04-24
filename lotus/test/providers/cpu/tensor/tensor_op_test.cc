#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/cpu/tensor/cast_op.h"

namespace Lotus {
namespace Test {

TEST(TensorOpTest, Reshape) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2});
  test.AddOutput<float>("reshaped", {1, 3, 2}, std::vector<float>(6, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Identity) {
  OpTester test("Identity");
  std::vector<float> X{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  test.AddInput<float>("input", {2, 3}, X);
  test.AddOutput<float>("output", {2, 3}, X);
  test.Run();
}

TEST(TensorOpTest, ShapeTest2D) {
  OpTester test("Shape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<int64_t>("shape", {2}, {2, 3});
  test.Run();
}

TEST(TensorOpTest, ShapeTest3D) {
  OpTester test("Shape");

  test.AddInput<float>("data", {2, 3, 4}, std::vector<float>(24, 1.0f));
  test.AddOutput<int64_t>("shape", {3}, {2, 3, 4});
  test.Run();
}

template <typename SrcType,
          typename DstType>
void TestCastOp(const std::initializer_list<SrcType> &input,
                const std::initializer_list<DstType> &output,
                const std::vector<int64_t> &dimensions,
                int64_t toType) {
  OpTester test("Cast");
  test.AddAttribute("to", toType);
  test.AddInput<SrcType>("input", dimensions, input);
  test.AddOutput<DstType>("output", dimensions, output);
  test.Run();
}

TEST(TensorOpTest, Cast) {
  auto input = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  const std::vector<int64_t> shape{3, 2, 2};

  auto float_output = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  TestCastOp(input, float_output, shape, TensorProto::FLOAT);

  auto double_output = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
  TestCastOp(input, double_output, shape, TensorProto::DOUBLE);

  auto bool_output = {false, true, true, true, true, true, true, true, true, true, true, true};
  TestCastOp(input, bool_output, shape, TensorProto::BOOL);

  const std::initializer_list<uint8_t> uint8_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, uint8_t_output, shape, TensorProto::UINT8);

  const std::initializer_list<uint16_t> uint16_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, uint16_t_output, shape, TensorProto::UINT16);

  const std::initializer_list<uint32_t> uint32_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, uint32_t_output, shape, TensorProto::UINT32);

  const std::initializer_list<uint64_t> uint64_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, uint64_t_output, shape, TensorProto::UINT64);

  const std::initializer_list<int16_t> int16_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, int16_t_output, shape, TensorProto::INT16);

  const std::initializer_list<int> int_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, int_output, shape, TensorProto::INT32);

  const std::initializer_list<int64_t> int64_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, int64_t_output, shape, TensorProto::INT64);
}

}  // namespace Test
}  // namespace Lotus

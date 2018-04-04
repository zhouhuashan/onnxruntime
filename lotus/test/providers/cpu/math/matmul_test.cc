#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

TEST(MathOpTest, MatMul) {
  std::vector<float> vals{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

  struct MatMulTest {
    std::vector<int64_t> input0_dims_;
    std::vector<int64_t> input1_dims_;
    std::vector<int64_t> expected_dims_;
    std::vector<float> expected_vals_;
  };

  MatMulTest testcases[] =
      {
          // test padding and broadcast
          {{3, 1, 1, 2},
           {2, 2, 2},
           {3, 2, 1, 2},
           {2, 3, 6, 7, 6, 11, 26, 31, 10, 19, 46, 55}},
          // test padding and broadcast
          {{2, 3, 2},
           {3, 2, 2, 1},
           {3, 2, 3, 1},
           {1, 3, 5, 33, 43, 53, 5, 23, 41, 85, 111, 137, 9, 43, 77, 137, 179, 221}},
          // test left 1D
          {{2},
           {3, 2, 2},
           {3, 2},
           {2, 3, 6, 7, 10, 11}},
          // test right 1D
          {{3, 2, 2},
           {2},
           {3, 2},
           {1, 3, 5, 7, 9, 11}},
          // test scalar output
          {{3},
           {3},
           {},
           {5}},
          // test 2D
          {{3, 4},
           {4, 3},
           {3, 3},
           {42, 48, 54, 114, 136, 158, 186, 224, 262}},
      };

  for (auto t : testcases) {
    OpTester test("MatMul");

    std::vector<float> input0_vals(vals.cbegin(), vals.cbegin() + TensorShape::SizeHelper(t.input0_dims_, 0, t.input0_dims_.size()));
    test.AddInput<float>("A", t.input0_dims_, input0_vals);

    std::vector<float> input1_vals(vals.cbegin(), vals.cbegin() + TensorShape::SizeHelper(t.input1_dims_, 0, t.input1_dims_.size()));
    test.AddInput<float>("B", t.input1_dims_, input1_vals);

    test.AddOutput<float>("Y", t.expected_dims_, t.expected_vals_);
    test.Run();
  }
}

}  // namespace Test
}  // namespace Lotus

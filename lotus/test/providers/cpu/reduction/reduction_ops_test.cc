#include "core/providers/cpu/reduction/reduction_ops.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/cpu/reduction/reduction_test_cases.h"

namespace Lotus {
namespace Test {

template <typename OutT>
void TestReduceOp(const std::string &op,
                  const std::vector<int64_t> &input_dims,
                  const std::vector<float> &data,
                  const std::vector<int64_t> &axes,
                  int64_t keepdims,
                  const std::vector<int64_t> &expected_dims,
                  const std::vector<OutT> &expected_data)

{
  OpTester test(op.c_str());
  if (!axes.empty())
  {
      if (op.compare("ArgMax") == 0 || op.compare("ArgMin") == 0)
          test.AddAttribute("axis", axes[0]);
      else
          test.AddAttribute("axes", axes);
  }
  test.AddAttribute("keepdims", keepdims);
  test.AddInput<float>("data", input_dims, data);
  test.AddOutput<OutT>("reduced", expected_dims, expected_data);
  test.Run();
}

TEST(ReductionOpTest, ReductionVariationTest) {
  const std::vector<float> &input_data = testcases.input_data;
  const std::vector<int64_t> &input_dims = testcases.input_dims;
  OpAttributesResultMap &opAttributesResultMap = testcases.map_op_attribute_expected;

  for (auto a : opAttributesResultMap) {
    const ReductionAttribute &attributes = std::get<0>(a.second);
    const std::vector<int64_t> expected_dims = std::get<1>(a.second);
    if (a.first.compare("ArgMax") == 0 || a.first.compare("ArgMin") == 0) {
      std::vector<int64_t> expected_values;
      for (auto v : std::get<2>(a.second))
        expected_values.push_back(static_cast<int64_t>(v));
      TestReduceOp<int64_t>(a.first, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                            expected_dims, expected_values);
    } else {
      const std::vector<float> expected_values = std::get<2>(a.second);
      TestReduceOp<float>(a.first, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                          expected_dims, expected_values);
    }
  }
}

TEST(ReductionOpTest, ReduceL1) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {33.0f, 45.0f});
  test.Run();
}  // namespace Test

TEST(ReductionOpTest, ReduceL2) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {2}, {15.71623325f, 20.07485962f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSum) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2},
                        {1.09861231f, 2.07944155f,
                         3.55534792f, 3.87120104f,
                         4.59511995f, 4.7874918f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {10.33174133f, 12.33174133f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 1}, {4.0f, 8.0f, 12.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {5.5f, 7.5f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {1.0f, 3.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {33.0f, 45.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {247.0f, 403.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {5400.f, 88704.f});
  test.Run();
}

TEST(ReductionOpTest, ArgMax) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run();
}

TEST(ReductionOpTest, ArgMin) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {0, 0,
                           0, 0});
  test.Run();
}

}  // namespace Test
}  // namespace Lotus

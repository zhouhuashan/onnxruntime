#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/cpu/tensor/cast_op.h"

namespace Lotus {
namespace Test {

typedef std::vector<LotusIR::NodeArg*> ArgMap;
TEST(TensorOpTest, Reshape) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2});
  test.AddOutput<float>("reshaped", {1, 3, 2}, std::vector<float>(6, 1.0f));
  test.Run();
}

template <typename SrcType,
    typename DstType>
    void TestCastOp(const std::vector<SrcType> &input,
        const std::vector<DstType> &output,
        const std::vector<int64_t> &dimensions,
        std::string toType) {
    OpTester test("Cast");

    test.AddAttribute("to", std::vector<std::string>{toType});
    test.AddInput<SrcType>("input", dimensions, input);
    test.AddOutput<DstType>("output", dimensions, output);
    test.Run();
}

TEST(TensorOpTest, Cast) {
    const std::vector<float> input{ 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f };
    const std::vector<int64_t> shape{ 3, 2, 2 };
    
    const std::vector<float> float_output{ 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f };
    TestCastOp<float, float>(input, float_output, shape, "FLOAT");

    const std::vector<double> double_output{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0 };
    TestCastOp<float, double>(input, double_output, shape, "DOUBLE");

    // Needs support at AddData()
    //const std::vector<bool> bool_output{ false, true, true, true, true, true, true, true, true, true, true, true };
    //TestCastOp<float, bool>(input, bool_output, shape, "BOOL");

    const std::vector<uint8_t> uint8_t_output{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    TestCastOp<float, uint8_t>(input, uint8_t_output, shape, "UINT8");

    const std::vector<uint16_t> uint16_t_output{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    TestCastOp<float, uint16_t>(input, uint16_t_output, shape, "UINT16");

    const std::vector<uint32_t> uint32_t_output{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    TestCastOp<float, uint32_t>(input, uint32_t_output, shape, "UINT32");

    const std::vector<uint64_t> uint64_t_output{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    TestCastOp<float, uint64_t>(input, uint64_t_output, shape, "UINT64");

    const std::vector<int16_t> int16_t_output{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    TestCastOp<float, int16_t>(input, int16_t_output, shape, "INT16");

    const std::vector<int> int_output{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    TestCastOp<float, int>(input, int_output, shape, "INT32");

    const std::vector<int64_t> int64_t_output{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    TestCastOp<float, int64_t>(input, int64_t_output, shape, "INT64");
}
}  // namespace Test
}  // namespace Lotus

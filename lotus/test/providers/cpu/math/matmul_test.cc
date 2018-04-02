#include "core/providers/cpu/math/matmul.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace Lotus {
namespace Test {

static const TypeProto_Set s_typeProto_float{TensorProto_DataType_FLOAT};

TEST(MathOpTest, MatMul) {
  LotusIR::NodeArg x_def("X", &s_typeProto_float),
      w_def("W", &s_typeProto_float),
      output_def("Y", &s_typeProto_float);
  std::vector<LotusIR::NodeArg*> input_defs{&x_def, &w_def};
  std::vector<LotusIR::NodeArg*> output_defs{&output_def};
  CREATE_NODE("MatMul", input_defs, output_defs);

  AllocatorInfo allocator_info("CPUAllocator", AllocatorType::kArenaAllocator);
  KernelDef kernel_def;
  OpKernelInfo info(*node, allocator_info, kernel_def);
  MatMul<float> kernel(info);

  std::vector<float> vals{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

  SessionState state;
  state.SetGraph(graph);
  SetupState(state, input_defs, output_defs);

  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(input_defs, output_defs, feeds, output_names);

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
    auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(state, feeds, output_names);

    std::vector<float> input0_vals(vals.cbegin(), vals.cbegin() + TensorShape::SizeHelper(t.input0_dims_, 0, t.input0_dims_.size()));
    auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, t.input0_dims_, &input0_vals);
    EXPECT_TRUE(status.IsOK());

    std::vector<float> input1_vals(vals.cbegin(), vals.cbegin() + TensorShape::SizeHelper(t.input1_dims_, 0, t.input1_dims_.size()));
    status = TestUtils::PrepareIthInput<float>(*node, 1, frame, t.input1_dims_, &input1_vals);
    EXPECT_TRUE(status.IsOK());

    status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, t.expected_dims_);
    EXPECT_TRUE(status.IsOK());

    OpKernelContext kernel_ctx(frame.get(), static_cast<OpKernel*>(&kernel), DefaultLoggingManager().DefaultLogger());
    kernel.Compute(&kernel_ctx);
    auto output = kernel_ctx.Output(0, TensorShape(t.expected_dims_));
    const float* res = output->Data<float>();

    for (int i = 0; i < t.expected_vals_.size(); ++i) {
      EXPECT_EQ(t.expected_vals_[i], res[i]);
    }
  }
}

}  // namespace Test
}  // namespace Lotus

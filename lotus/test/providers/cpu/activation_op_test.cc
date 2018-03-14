#include "core/providers/cpu/activation/relu.h"
#include "gtest/gtest.h"
#include "test/test_utils.h"

namespace Lotus {
    namespace Test {
        typedef  std::vector<LotusIR::NodeArg*> ArgMap;
        TEST(ActivationOpTest, ReLU) {
            CREATE_NODE(relu);
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);
            node->Mutable_InputDefs().push_back(&input_def);
            node->Mutable_OutputDefs().push_back(&output_def);

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            OpKernelInfo info(*node, allocator_info);
            ReLU<float> kernel(&info, nullptr);
            ExecutionFrame frame;

            std::vector<float> input_vals = {
                -1.0f, 0, 1.0f, // normal input values for activation
                FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10, // min, denorm, -denorm
                FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity() }; // max, -max, inf
            std::vector<int64_t> dims = { 3, 3 };
            std::vector<float> expected_vals;
            for (auto f : input_vals)
                expected_vals.push_back(std::max(0.0f, f));
            auto input = TestUtils::CreateTensor<float>(dims, input_vals);
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3 * 3));
            auto ctx = TestUtils::CreateKernelContext(&kernel, frame, input.get(), output.get());
            kernel.compute(ctx.get());
            const float* res = output->data<float>();

            for (int i = 0; i < expected_vals.size(); ++i) {
                EXPECT_EQ(expected_vals[i], res[i]);
            }
        }
    }
}

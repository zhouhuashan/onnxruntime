#include "core/providers/cpu/math/clip.h"
#include "gtest/gtest.h"
#include "test/test_utils.h"

namespace Lotus {
    namespace Test {
        typedef  std::vector<LotusIR::NodeArg*> ArgMap;
        TEST(MathOpTest, Clip) {
            CREATE_NODE(clip);
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);
            node->Mutable_InputDefs().push_back(&input_def);
            node->Mutable_OutputDefs().push_back(&output_def);
            EXPECT_TRUE(node->AddAttribute("min", -10.0f));
            EXPECT_TRUE(node->AddAttribute("max", 10.0f));

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            OpKernelInfo info(*node, allocator_info);
            Clip<float> kernel(&info);
            ExecutionFrame frame;
            
            std::vector<float> input_vals = { 11.0f, 4.4f, 432.3f, -1.3f, 3.5f, 64.0f, -5.4f, 9.3f, 82.4f };
            std::vector<int64_t> dims = { 3, 3 };
            std::vector<float> expected_vals = { 10.0f, 4.4f, 10.0f, -1.3f, 3.5f, 10.0f, -5.4f, 9.3f, 10.0f };
            auto input = TestUtils::CreateTensor<float>(dims, input_vals);
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            auto ctx = TestUtils::CreateKernelContext(&kernel, frame, input.get(), output.get());
            kernel.compute(ctx.get());
            const float* res = output->data<float>();
            
            for (int i = 0; i < expected_vals.size(); ++i) {
                EXPECT_EQ(expected_vals[i], res[i]);
            }
        }
    }
}

#include "core/providers/cpu/math/clip.h"
#include "gtest/gtest.h"
#include "test/test_utils.h"

namespace Lotus {
    namespace Test {
        typedef  std::vector<LotusIR::NodeArg*> ArgMap;
        TEST(MathOpTest, Clip) {
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);
            std::vector<LotusIR::NodeArg*> input_defs{ &input_def };
            std::vector<LotusIR::NodeArg*> output_defs{ &output_def };
            CREATE_NODE(Clip, input_defs, output_defs);
            
            EXPECT_TRUE(node->AddAttribute("min", -10.0f));
            EXPECT_TRUE(node->AddAttribute("max", 10.0f));

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            KernelDef kernel_def;
            OpKernelInfo info(*node, allocator_info, kernel_def);
            Clip<float> kernel(info);

            std::vector<float> input_vals = { 11.0f, 4.4f, 432.3f, -1.3f, 3.5f, 64.0f, -5.4f, 9.3f, 82.4f };
            std::vector<int64_t> dims = { 3, 3 };
            std::vector<float> expected_vals = { 10.0f, 4.4f, 10.0f, -1.3f, 3.5f, 10.0f, -5.4f, 9.3f, 10.0f };

            SessionState state;
            auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(graph, state);
            auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, dims, &input_vals);
            EXPECT_TRUE(status.IsOK());
            status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, dims);
            EXPECT_TRUE(status.IsOK());

            OpKernelContext kernel_ctx(frame.get(), static_cast<OpKernel*>(&kernel));
            kernel.compute(&kernel_ctx);
            auto output = kernel_ctx.output(0, TensorShape(dims));
            const float* res = output->data<float>();
            
            for (int i = 0; i < expected_vals.size(); ++i) {
                EXPECT_EQ(expected_vals[i], res[i]);
            }
        }
    }
}

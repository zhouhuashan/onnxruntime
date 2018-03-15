#include "core/providers/cpu/activation/sigmoid.h"
#include "core/providers/cpu/activation/tanh.h"
#include "core/providers/cpu/activation/relu.h"
#include "gtest/gtest.h"
#include "test/test_utils.h"

namespace Lotus {
    namespace Test {
        typedef  std::vector<LotusIR::NodeArg*> ArgMap;

        TEST(ActivationOpTest, Sigmoid) {
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);
            std::vector<LotusIR::NodeArg*> input_defs{ &input_def };
            std::vector<LotusIR::NodeArg*> output_defs{ &output_def };
            CREATE_NODE(Sigmoid, input_defs, output_defs);

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            KernelDef kernel_def;
            OpKernelInfo info(*node, allocator_info, kernel_def);

            Sigmoid<float> kernel(info);
            SessionState state;
            auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(graph, state);

            std::vector<float> input_vals = { -3.f, -2.f, -1.f, 1.f, 2.f, 3.f };
            std::vector<int64_t> dims = { 2, 3 };

            auto s = [](float x) {
                auto y = 1.f / (1.f + std::exp(-x));
                return y;
            };

            std::vector<float> expected_vals = { s(-3.f), s(-2.f), s(-1.f), s(1.f), s(2.f), s(3.f) };

            auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, dims, &input_vals);
            EXPECT_TRUE(status.IsOK());
            status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, dims);
            EXPECT_TRUE(status.IsOK());

            OpKernelContext ctx(frame.get(), static_cast<OpKernel*>(&kernel));
            kernel.compute(&ctx);
            auto output = ctx.output(0, TensorShape(dims));
            const float* res = output->data<float>();

            for (int i = 0; i < expected_vals.size(); ++i) {
                EXPECT_FLOAT_EQ(expected_vals[i], res[i]);
            }
        }

        TEST(ActivationOpTest, Tanh) {
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);
            std::vector<LotusIR::NodeArg*> input_defs{ &input_def };
            std::vector<LotusIR::NodeArg*> output_defs{ &output_def };
            CREATE_NODE(Tanh, input_defs, output_defs);

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            KernelDef kernel_def;
            OpKernelInfo info(*node, allocator_info, kernel_def);

            Tanh<float> kernel(info);
            SessionState state;
            auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(graph, state);

            std::vector<float> input_vals = { -3.f, -2.f, -1.f, 1.f, 2.f, 3.f };
            std::vector<int64_t> dims = { 2, 3 };

            auto s = [](float x) {
                auto y = std::tanh(x);
                return y;
            };

            std::vector<float> expected_vals = { s(-3.f), s(-2.f), s(-1.f), s(1.f), s(2.f), s(3.f) };

            auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, dims, &input_vals);
            EXPECT_TRUE(status.IsOK());
            status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, dims);
            EXPECT_TRUE(status.IsOK());

            OpKernelContext ctx(frame.get(), static_cast<OpKernel*>(&kernel));
            kernel.compute(&ctx);
            auto output = ctx.output(0, TensorShape(dims));
            const float* res = output->data<float>();

            for (int i = 0; i < expected_vals.size(); ++i) {
                EXPECT_FLOAT_EQ(expected_vals[i], res[i]);
            }
        }

        TEST(ActivationOpTest, ReLU) {
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);
            std::vector<LotusIR::NodeArg*> input_defs{ &input_def };
            std::vector<LotusIR::NodeArg*> output_defs{ &output_def };
            CREATE_NODE(Relu, input_defs, output_defs);

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            KernelDef kernel_def;
            OpKernelInfo info(*node, allocator_info, kernel_def);

            ReLU<float> kernel(info);
            SessionState state;
            auto frame = TestUtils::CreateSingleNodeCPUExecutionFrame(graph, state);

            std::vector<float> input_vals = {
                -1.0f, 0, 1.0f, // normal input values for activation
                FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10, // min, denorm, -denorm
                FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity() }; // max, -max, inf
            std::vector<int64_t> dims = { 3, 3 };
            std::vector<float> expected_vals;
            for (auto f : input_vals)
                expected_vals.push_back(std::max(0.0f, f));
            auto status = TestUtils::PrepareIthInput<float>(*node, 0, frame, dims, &input_vals);
            EXPECT_TRUE(status.IsOK());
            status = TestUtils::PrepareIthOutput<float>(*node, 0, frame, dims);
            EXPECT_TRUE(status.IsOK());

            OpKernelContext ctx(frame.get(), static_cast<OpKernel*>(&kernel));
            kernel.compute(&ctx);
            auto output = ctx.output(0, TensorShape(dims));
            const float* res = output->data<float>();

            for (int i = 0; i < expected_vals.size(); ++i) {
                EXPECT_EQ(expected_vals[i], res[i]);
            }
        }
    }
}

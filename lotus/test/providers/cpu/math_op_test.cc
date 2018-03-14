#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/math/clip.h"
#include "gtest/gtest.h"
#include "test/test_utils.h"

namespace Lotus {
    namespace Test {
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
        
        struct SimpleFloatTest
        {
            SimpleFloatTest() {
                tensor_float_.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            }

            template<typename Op>
            void Run(std::vector<Tensor*> inputs, std::vector<Tensor*> outputs) {
                LotusIR::Node* node = graph_->GetNode(graph_->NumberOfNodes() - 1);

                AllocatorInfo allocator_info{"CPUAllocator", Lotus::AllocatorType::ArenaAllocator};
                OpKernelInfo info(*node, allocator_info);
                Op kernel(&info);
                ExecutionFrame frame;
                auto ctx = TestUtils::CreateKernelContext(&kernel, frame, std::move(inputs), std::move(outputs));
                kernel.compute(ctx.get());
            }

            template<size_t count>
            static void Check(Tensor& output, const float (&expected_vals)[count]) {
                LOTUS_ENFORCE(output.shape().Size()==count);
                const float* res = output.data<float>();
                for (int i = 0; i < count; ++i) {
                    EXPECT_NEAR(expected_vals[i], res[i], 0.001f);
                }
            }

            LotusIR::Model model_{"test"};
            LotusIR::Graph* graph_{model_.MainGraph()};
            TypeProto tensor_float_;
        };

        TEST(MathOpTest, Add) {
            SimpleFloatTest test;
            LotusIR::NodeArg input1_def("A", &test.tensor_float_), input2_def("B", &test.tensor_float_), output_def("C", &test.tensor_float_);
            test.graph_->AddNode("node1", "add", "add operator", {&input1_def, &input2_def}, {&output_def});

            std::vector<int64_t> dims{ 3, 3 };
            auto input1 = TestUtils::CreateTensor<float>(dims, {  1.0f, 2.0f,  -1.0f, 0.0f, 1.5f, -100.0f,  -5.4f,  9.3f, -10'000.0f });
            auto input2 = TestUtils::CreateTensor<float>(dims, { -1.0f, 4.4f, 432.3f, 0.0f, 3.5f,   64.0f,  -5.4f,  9.3f,  10'000.0f });
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            float expected_vals[] =                            {  0.0f, 6.4f, 431.3f, 0.0f, 5.0f,  -36.0f, -10.8f, 18.6f,       0.0f };

            test.Run<Add<float>>({ input1.get(), input2.get() }, { output.get() });
            test.Check(*output, expected_vals);
        }

        TEST(MathOpTest, Sum) {
            SimpleFloatTest test;
            LotusIR::NodeArg input1_def("A", &test.tensor_float_), input2_def("B", &test.tensor_float_), input3_def("A3", &test.tensor_float_), output_def("C", &test.tensor_float_);
            test.graph_->AddNode("node1", "sum", "sum operator", {&input1_def, &input2_def, &input3_def}, {&output_def});

            std::vector<int64_t> dims{ 3, 3 };
            auto input1 = TestUtils::CreateTensor<float>(dims, {  1.0f, 0.0f, 1.0f, -1.0f, 1.1f,  -100.0f,  -5.4f,  0.01f, -10'000.0f });
            auto input2 = TestUtils::CreateTensor<float>(dims, {  1.0f, 0.0f, 2.0f, -2.0f, 2.2f,    64.0f,  -1.0f,  0.02f,       0.1f });
            auto input3 = TestUtils::CreateTensor<float>(dims, {  1.0f, 0.0f, 3.0f, -3.0f, 3.3f,    64.0f,   5.4f,  0.03f,  10'000.0f });
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            float expected_vals[] =                            {  3.0f, 0.0f, 6.0f, -6.0f, 6.6f,    28.0f,  -1.0f,  0.06f,       0.1f };

            test.Run<Sum<float>>({ input1.get(), input2.get(), input3.get() }, { output.get() });
            test.Check(*output, expected_vals);
        }

        TEST(MathOpTest, Sub) {
            SimpleFloatTest test;
            LotusIR::NodeArg input1_def("A", &test.tensor_float_), input2_def("B", &test.tensor_float_), output_def("C", &test.tensor_float_);
            test.graph_->AddNode("node1", "sub", "sub operator", {&input1_def, &input2_def}, {&output_def});
            
            std::vector<int64_t> dims{ 3, 3 };
            auto input1 = TestUtils::CreateTensor<float>(dims, {  1.0f, 2.0f,  -1.0f, 0.0f, 1.5f, -100.0f, -5.4f,  9.3f, -10'000.0f });
            auto input2 = TestUtils::CreateTensor<float>(dims, { -1.0f, 4.4f, 432.3f, 0.0f, 3.5f,   64.0f, -5.4f,  9.3f,  10'000.0f });
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            float expected_vals[] =                            {  2.0f,-2.4f,-433.3f, 0.0f,-2.0f, -164.0f,  0.0f,  0.0f, -20'000.0f };

            test.Run<Sub<float>>({ input1.get(), input2.get() }, { output.get() });
            test.Check(*output, expected_vals);
        }

        TEST(MathOpTest, Mul) {
            SimpleFloatTest test;
            LotusIR::NodeArg input1_def("A", &test.tensor_float_), input2_def("B", &test.tensor_float_), output_def("C", &test.tensor_float_);
            test.graph_->AddNode("node1", "mul", "mul operator", {&input1_def, &input2_def}, {&output_def});

            std::vector<int64_t> dims{ 3, 3 };
            auto input1 = TestUtils::CreateTensor<float>(dims, {  1.0f, 2.0f,  -1.0f, 0.0f, 1.5f,   -100.0f,  -5.4f,  9.30f,      -10'000.0f });
            auto input2 = TestUtils::CreateTensor<float>(dims, { -1.0f, 4.4f, 432.3f, 0.0f, 3.5f,     64.0f,  -5.4f,  9.30f,       10'000.0f });
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            float expected_vals[] =                            { -1.0f, 8.8f,-432.3f, 0.0f, 5.25f,-6'400.0f, 29.16f, 86.49f, -100'000'000.0f };

            test.Run<Mul<float>>({ input1.get(), input2.get() }, { output.get() });
            test.Check(*output, expected_vals);
        }

        TEST(MathOpTest, Reciprocal) {
            SimpleFloatTest test;
            LotusIR::NodeArg input_def("X", &test.tensor_float_), output_def("Y", &test.tensor_float_);
            test.graph_->AddNode("node1", "reciprocal", "reciprocal operator", {&input_def}, {&output_def});

            std::vector<int64_t> dims{ 2, 2 };
            auto input  = TestUtils::CreateTensor<float>(dims, {  1.0f, 2.0f, -1.0f, -2.0f });
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            float expected_vals[] =                            {  1.0f, 0.5f, -1.0f, -0.5f };

            test.Run<Reciprocal<float>>({ input.get() }, { output.get() });
            test.Check(*output, expected_vals);
        }

    }
}

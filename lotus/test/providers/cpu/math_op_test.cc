#include "core/kernels/cpu/math/clip.h"
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

        TEST(MathOpTest, Constant) {
            LotusIR::Model model("test");
            LotusIR::Graph* graph = model.MainGraph();
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg output_def("C", &tensor_float);

            graph->AddNode("node1", "constant", "constant operator", {}, {&output_def});
            LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            OpKernelInfo info(*node, allocator_info);
            Constant<float> kernel(&info);
            ExecutionFrame frame;

            std::vector<int64_t> dims{ 2, 3 };
            auto input1 = TestUtils::CreateTensor<float>(dims, {  11.0f, 12.0f, 13.0f, 21.0f, 22.0f, 23.0f });
//            EXPECT_TRUE(node->AddAttribute("constant", input1.get()));

            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(2*3));
            float expected_vals[] =                            {  11.0f, 12.0f, 13.0f, 21.0f, 22.0f, 23.0f };
            auto ctx = TestUtils::CreateKernelContext(&kernel, frame, {}, { output.get() });
            kernel.compute(ctx.get());
            const float* res = output->data<float>();
            
            for (int i = 0; i < _countof(expected_vals); ++i) {
                EXPECT_EQ(expected_vals[i], res[i]);
            }
        }

        TEST(MathOpTest, Add) {
            LotusIR::Model model("test");
            LotusIR::Graph* graph = model.MainGraph();
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input1_def("A", &tensor_float), input2_def("B", &tensor_float), output_def("C", &tensor_float);

            graph->AddNode("node1", "add", "add operator", {&input1_def, &input2_def}, {&output_def});
            LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            OpKernelInfo info(*node, allocator_info);
            Add<float> kernel(&info);
            ExecutionFrame frame;
            
            std::vector<int64_t> dims{ 3, 3 };
            auto input1 = TestUtils::CreateTensor<float>(dims, {  1.0f, 2.0f,  -1.0f, 0.0f, 1.5f, -100.0f,  -5.4f,  9.3f, -10000.0f });
            auto input2 = TestUtils::CreateTensor<float>(dims, { -1.0f, 4.4f, 432.3f, 0.0f, 3.5f,   64.0f,  -5.4f,  9.3f,  10000.0f });
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            float expected_vals[] =                            {  0.0f, 6.4f, 431.3f, 0.0f, 5.0f,  -36.0f, -10.8f, 18.6f,      0.0f };
            auto ctx = TestUtils::CreateKernelContext(&kernel, frame, { input1.get(), input2.get() }, { output.get() });
            kernel.compute(ctx.get());
            const float* res = output->data<float>();
            
            for (int i = 0; i < _countof(expected_vals); ++i) {
                EXPECT_EQ(expected_vals[i], res[i]);
            }
        }

        TEST(MathOpTest, Sum) {
            LotusIR::Model model("test");
            LotusIR::Graph* graph = model.MainGraph();
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input1_def("A", &tensor_float), input2_def("B", &tensor_float), input3_def("A3", &tensor_float), output_def("C", &tensor_float);

            graph->AddNode("node1", "sum", "sum operator", {&input1_def, &input2_def, &input3_def}, {&output_def});
            LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            OpKernelInfo info(*node, allocator_info);
            Sum<float> kernel(&info);
            ExecutionFrame frame;

            std::vector<int64_t> dims{ 3, 3 };
            auto input1 = TestUtils::CreateTensor<float>(dims, {  1.0f, 0.0f, 1.0f, -1.0f, 1.1f,  -100.0f,  -5.4f,  0.01f, -10'000.0f });
            auto input2 = TestUtils::CreateTensor<float>(dims, {  1.0f, 0.0f, 2.0f, -2.0f, 2.2f,    64.0f,  -1.0f,  0.02f,       0.1f });
            auto input3 = TestUtils::CreateTensor<float>(dims, {  1.0f, 0.0f, 3.0f, -3.0f, 3.3f,    64.0f,   5.4f,  0.03f,  10'000.0f });
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            float expected_vals[] =                            {  3.0f, 0.0f, 6.0f, -6.0f, 6.6f,    28.0f,  -1.0f,  0.06f,       0.1f };
            auto ctx = TestUtils::CreateKernelContext(&kernel, frame, { input1.get(), input2.get(), input3.get() }, { output.get() });
            kernel.compute(ctx.get());
            const float* res = output->data<float>();

            for (int i = 0; i < _countof(expected_vals); ++i) {
                EXPECT_NEAR(expected_vals[i], res[i], 0.001f);
            }
        }


        TEST(MathOpTest, Sub) {
            LotusIR::Model model("test");
            LotusIR::Graph* graph = model.MainGraph();
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input1_def("A", &tensor_float), input2_def("B", &tensor_float), output_def("C", &tensor_float);

            graph->AddNode("node1", "sub", "sub operator", {&input1_def, &input2_def}, {&output_def});
            LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            OpKernelInfo info(*node, allocator_info);
            Sub<float> kernel(&info);
            ExecutionFrame frame;
            
            std::vector<int64_t> dims{ 3, 3 };
            auto input1 = TestUtils::CreateTensor<float>(dims, {  1.0f, 2.0f,  -1.0f, 0.0f, 1.5f, -100.0f, -5.4f,  9.3f, -10000.0f });
            auto input2 = TestUtils::CreateTensor<float>(dims, { -1.0f, 4.4f, 432.3f, 0.0f, 3.5f,   64.0f, -5.4f,  9.3f,  10000.0f });
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            float expected_vals[] =                            {  2.0f,-2.4f,-433.3f, 0.0f,-2.0f, -164.0f,  0.0f,  0.0f, -20000.0f };
            auto ctx = TestUtils::CreateKernelContext(&kernel, frame, { input1.get(), input2.get() }, { output.get() });
            kernel.compute(ctx.get());
            const float* res = output->data<float>();
            
            for (int i = 0; i < _countof(expected_vals); ++i) {
                EXPECT_EQ(expected_vals[i], res[i]);
            }
        }

        TEST(MathOpTest, Mul) {
            LotusIR::Model model("test");
            LotusIR::Graph* graph = model.MainGraph();
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input1_def("A", &tensor_float), input2_def("B", &tensor_float), output_def("C", &tensor_float);

            graph->AddNode("node1", "mul", "mul operator", {&input1_def, &input2_def}, {&output_def});
            LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            OpKernelInfo info(*node, allocator_info);
            Mul<float> kernel(&info);
            ExecutionFrame frame;
            
            std::vector<int64_t> dims{ 3, 3 };
            auto input1 = TestUtils::CreateTensor<float>(dims, {  1.0f, 2.0f,  -1.0f, 0.0f, 1.5f,   -100.0f,  -5.4f,  9.30f,      -10'000.0f });
            auto input2 = TestUtils::CreateTensor<float>(dims, { -1.0f, 4.4f, 432.3f, 0.0f, 3.5f,     64.0f,  -5.4f,  9.30f,       10'000.0f });
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            float expected_vals[] =                            { -1.0f, 8.8f,-432.3f, 0.0f, 5.25f, 6'400.0f, 29.16f, 86.49f, -100'000'000.0f };
            auto ctx = TestUtils::CreateKernelContext(&kernel, frame, { input1.get(), input2.get() }, { output.get() });
            kernel.compute(ctx.get());
            const float* res = output->data<float>();
            
            for (int i = 0; i < _countof(expected_vals); ++i) {
                EXPECT_EQ(expected_vals[i], res[i]);
            }
        }

        TEST(MathOpTest, Concat) {
            LotusIR::Model model("test");
            LotusIR::Graph* graph = model.MainGraph();
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            LotusIR::NodeArg input1_def("A", &tensor_float), input2_def("B", &tensor_float), input3_def("A3", &tensor_float), output_def("C", &tensor_float);

            graph->AddNode("node1", "concat", "sum operator", {&input1_def, &input2_def, &input3_def}, {&output_def});
            LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);

            EXPECT_TRUE(node->AddAttribute("axis", 1));

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            OpKernelInfo info(*node, allocator_info);
            Sum<float> kernel(&info);
            ExecutionFrame frame;

            std::vector<int64_t> dims{ 3, 3 };
            auto input1 = TestUtils::CreateTensor<float>(dims, {  1.0f, 0.0f, 1.0f, -1.0f, 1.1f,  -100.0f,  -5.4f,  0.01f, -10'000.0f });
            auto input2 = TestUtils::CreateTensor<float>(dims, {  1.0f, 0.0f, 2.0f, -2.0f, 2.2f,    64.0f,  -1.0f,  0.02f,       0.1f });
            auto input3 = TestUtils::CreateTensor<float>(dims, {  1.0f, 0.0f, 3.0f, -3.0f, 3.3f,    64.0f,   5.4f,  0.03f,  10'000.0f });
            auto output = TestUtils::CreateTensor<float>(dims, std::vector<float>(3*3));
            float expected_vals[] =                            {  3.0f, 0.0f, 6.0f, -6.0f, 6.6f,    28.0f,  -1.0f,  0.06f,       0.1f };
            auto ctx = TestUtils::CreateKernelContext(&kernel, frame, { input1.get(), input2.get(), input3.get() }, { output.get() });
            kernel.compute(ctx.get());
            const float* res = output->data<float>();

            for (int i = 0; i < _countof(expected_vals); ++i) {
                EXPECT_NEAR(expected_vals[i], res[i], 0.001f);
            }
        }
    }
}

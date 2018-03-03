#ifdef _MSC_VER
#pragma warning(push)
// 'identifier' : unreferenced formal parameter
#pragma warning(disable: 4100)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable: 4800)
#endif
#include "google/protobuf/util/message_differencer.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <iostream>
#include "gtest/gtest.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"

namespace Lotus
{
    namespace Test
    {
        class TestOpKernel : public OpKernel {
        public:
            TestOpKernel(OpKernelInfo* p) : OpKernel(p) {}
            void compute(OpKernelContext* context) {
                UNUSED_PARAMETER(context);
            }
            void compute_async(OpKernelContext* context, DoneCallback done) {
                UNUSED_PARAMETER(context);
            }
        };

        TEST(SessionStateTest, AddGetKernelTest)
        {
            using google::protobuf::util::MessageDifferencer;

            SessionState s{ 10 }; // dummy

            LotusIR::Model model("graph_1");
            auto graph = model.MainGraph();
            std::vector<LotusIR::NodeArg*> inputs;
            std::vector<LotusIR::NodeArg*> outputs;
            TypeProto outputType;
            outputType.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            outputType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            LotusIR::NodeArg outputArg("node_1_out_1", &outputType);
            outputs.push_back(&outputArg);
            LotusIR::Node* p_node = graph->AddNode("node_1", "Variable", "node 1.", inputs, outputs);

            AllocatorInfo allocator_info("CPUAllocator", Lotus::AllocatorType::ArenaAllocator);
            OpKernelInfo* p_info = new OpKernelInfo(*p_node, allocator_info);
            unique_ptr<TestOpKernel> p_kernel;
            p_kernel.reset(new TestOpKernel(p_info));
            size_t orig_num_outputs = p_kernel->node().OutputDefs().size();
            //std::cout << "node_idx: " << p_node->Index() << std::endl;
            s.AddKernel(p_node->Index(), std::move(p_kernel));
            OpKernel* test_kernel = s.GetKernel(p_node->Index());
            //std::cout << "orig: " << orig_num_outputs << " new: " << test_kernel->num_outputs() << std::endl;
            EXPECT_EQ(orig_num_outputs, test_kernel->node().OutputDefs().size());
        }
    }
}

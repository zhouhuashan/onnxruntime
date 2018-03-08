#ifndef TEST_TEST_UTILS_H
#define TEST_TEST_UTILS_H

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"

namespace Lotus {
    namespace  Test {

        class TestUtils {
            typedef std::shared_ptr<LotusIR::Node> NodePtr;
            typedef std::shared_ptr<Tensor> TensorPtr;
            typedef std::shared_ptr<OpKernelContext> KernelContextPtr;


        public:

            static KernelContextPtr CreateKernelContext(OpKernel* kernel, ExecutionFrame& frame,
                Tensor* input, Tensor* output) {
                std::vector<Tensor*> inputs = { input };
                std::vector<Tensor*> outputs = { output };
                return CreateKernelContext(kernel, frame, inputs, outputs);
            }

            static KernelContextPtr CreateKernelContext(OpKernel* kernel,
                ExecutionFrame& frame, std::vector<Tensor*> inputs, std::vector<Tensor*> outputs) {
                //This a hack to facilitate the test due to our execution_frame is not 
                //fully implementated. Once the implementation is in place, we will replace 
                //this part of code with execution_frame func calls.

                for (Tensor* input : inputs) {
                    MLValue ml_input;
                    ml_input.pData = input;
                    ml_input.type = input->dtype();
                    frame.all_values_.push_back(ml_input);
                }
                for (Tensor* output : outputs) {
                    MLValue ml_output;
                    ml_output.pData = output;
                    ml_output.type = output->dtype();
                    frame.all_values_.push_back(ml_output);
                }
                for (int i = 0; i < frame.all_values_.size(); ++i) {
                    frame.node_values_.push_back(&frame.all_values_[i]);
                }

                ExecutionFrame::NodeInfo src_node, sink_node, test_node;
                test_node.kernel = kernel;
                frame.node_infos_.push_back(src_node);
                frame.node_infos_.push_back(sink_node);
                frame.node_infos_.push_back(test_node);
                
                OpKernelContext* ctx = new OpKernelContext(&frame, kernel);
                return std::shared_ptr<OpKernelContext>(ctx);
            }

            template<typename T>
            static TensorPtr CreateTensor(const std::vector<int64_t> dims, const std::vector<T> vals) {
                TensorShape shape(dims);
                size_t size = sizeof(T) * shape.Size();
                auto& alloc = AllocatorManager::Instance()->GetArena(CPU);
                auto data = alloc.Alloc(size);
                LOTUS_ENFORCE(data);
                memcpy(data, vals.data(), size);
                return std::make_shared<Tensor>(DataTypeImpl::GetType<T>(), shape, data, alloc.Info());
            }
        };

        #define CREATE_NODE(op_name)                                           \
          LotusIR::Model model("test");                                                     \
          LotusIR::Graph* graph = model.MainGraph();                                        \
          graph->AddNode("node1", #op_name, #op_name, ArgMap{}, ArgMap{});      \
          LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);
    }
}



#endif // !CORE_TEST_TEST_UTIL_H

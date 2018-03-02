#ifndef TEST_TEST_UTILS_H
#define TEST_TEST_UTILS_H

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_util.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"


//#include "gtest/gtest.h"


namespace Lotus {
    namespace  Test {

        class TestUtils{
            typedef std::shared_ptr<LotusIR::Node> NodePtr;
            typedef std::shared_ptr<Tensor> TensorPtr;
            typedef std::shared_ptr<OpKernelContext> KernelContextPtr;


        public:

            static KernelContextPtr CreateKernelContext(const LotusIR::Node& node, OpKernel* kernel, ExecutionFrame& frame,
                Tensor* input, Tensor* output) {
                std::vector<Tensor*> inputs={ input };
                std::vector<Tensor*> outputs={ output };
                return CreateKernelContext(node, kernel, frame, inputs, outputs);
            }
            
            static KernelContextPtr CreateKernelContext(const LotusIR::Node& node, OpKernel* kernel, 
                ExecutionFrame& frame, std::vector<Tensor*> inputs, std::vector<Tensor*> outputs) {
                //This a hack to facilitate the test due to our execution_frame is not 
                //fully implementated. Once the implementation is in place, we will replace 
                //this part of code with execution_frame func calls.

                for (Tensor* input : inputs) {
                    MLValue ml_input;
                    ml_input.pData = input;
                    ml_input.type = input->dtype();
                    frame.m_all_values.push_back(ml_input);
                }
                for (Tensor* output : outputs) {
                    MLValue ml_output;
                    ml_output.pData = output;
                    ml_output.type = output->dtype();
                    frame.m_all_values.push_back(ml_output);
                }
                for (int i = 0; i < frame.m_all_values.size(); ++i) {
                    frame.m_node_values.push_back(&frame.m_all_values[i]);
                }
                
                ExecutionFrame::NodeInfo src_node, sink_node, test_node;
                test_node.kernel = kernel;
                frame.m_node_infos.push_back(src_node);
                frame.m_node_infos.push_back(sink_node);
                frame.m_node_infos.push_back(test_node);
                OpKernelContext* ctx = new OpKernelContext(node, kernel, &frame);
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

    }
}



#endif // !CORE_TEST_TEST_UTIL_H

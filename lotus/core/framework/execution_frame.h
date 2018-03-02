#ifndef CORE_FRAMEWORK_EXECUTION_FRAME_H
#define CORE_FRAMEWORK_EXECUTION_FRAME_H

#include <mutex>
#include <vector>
#include "core/framework/ml_value.h"
#include "core/graph/graph.h"
#include "core/common/status.h"
#include "core/framework/tensor.h"

namespace Lotus
{
  namespace Test
  {
      class TestUtils;
  }

  class OpKernel;
  
  class ExecutionFrame {
  public:
      typedef MLValue* NodeArgValue;
      typedef std::vector<NodeArgValue> ArgTable;
    
      ExecutionFrame() {

      }

      ~ExecutionFrame() {

      }

        // Index to the first argument of the given node.
        int get_first_arg(const LotusIR::Node& node) {
            return m_node_infos[node.Index()].start_index;
        }

        template<typename T>
        const T* GetInput(int index) const {
            auto value = m_node_values[index];
            return reinterpret_cast<T*>(value->pData);
        }

        template<typename T>
        T* GetOutput(int index) {
            auto value = m_node_values[index];
            return reinterpret_cast<T*>(value->pData);
        }

    private:
        friend class OpKernelContext;

        //The TestUtils need hack this class to provide input/output 
        // tensors since the class is not fully implemented yet.
        friend class Lotus::Test::TestUtils;

        struct NodeInfo {
            // The kernel for this node.
            OpKernel* kernel = nullptr;

            // node_values_[start_index] is the first argument of this node.
            int start_index = 0;
        };

        Tensor* GetOrCreateTensor(int tensor_index, const TensorShape& shape) {
            // TODO:
            // Check whether the tensor has been allocated yet or not.
            // if it's allocated, then check the size of the allocated tensor with given shape,
            // if they match each other, then return, else throw error.
            // if it's not allocated, then allocate it with given shape and return.
            (tensor_index);
            (shape);
            return nullptr;
        }

        std::mutex m_mu;
        Status m_status;

        // The values for the inputs and outputs of the nodes.
        ArgTable m_node_values;

        // All the values for the entire graph.
        std::vector<MLValue> m_all_values;

        // The start index into node_values_ for all the nodes.
        std::vector<NodeInfo> m_node_infos;

        // i-th kernel is still waiting for pending_counts_[i] inputs.
        std::vector<int> m_pending_counts;
    };
}

#endif  // CORE_FRAMEWORK_EXECUTION_FRAME_H

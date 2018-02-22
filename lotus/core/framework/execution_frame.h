#ifndef CORE_FRAMEWORK_EXECUTION_FRAME_H
#define CORE_FRAMEWORK_EXECUTION_FRAME_H

#include <mutex>
#include <vector>
#include "core/framework/ml_value.h"
#include "core/graph/graph.h"
#include "core/common/status.h"

namespace Lotus
{
  class OpKernel;
  
  class ExecutionFrame {
  public:
    typedef MLValue* NodeArgValue;
    
    ExecutionFrame() {

    }

    ~ExecutionFrame() {

    }

    // Pointer to the first argument of the given node.
    NodeArgValue* get_first_arg(const LotusIR::Node& node) {
      int start_index = m_node_infos[node.Index()].start_index;
      return m_node_values.data() + start_index;
    }

  private:
    struct NodeInfo {
      // The kernel for this node.
      OpKernel* kernel = nullptr;

      // node_values_[start_index] is the first argument of this node.
      int start_index = 0;
    };

    std::mutex m_mu;
    Status m_status;

    // The values for the inputs and outputs of the nodes.
    std::vector<NodeArgValue> m_node_values;
    
    // All the values for the entire graph.
    std::vector<MLValue> m_all_values;

    // The start index into node_values_ for all the nodes.
    std::vector<NodeInfo> m_node_infos;

    // i-th kernel is still waiting for pending_counts_[i] inputs.
    std::vector<int> m_pending_counts;
  };
}

#endif  // CORE_FRAMEWORK_EXECUTION_FRAME_H

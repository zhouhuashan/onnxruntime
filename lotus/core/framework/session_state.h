#ifndef LOTUS_CORE_FRAMEWORK_SESSION_STATE_H_
#define LOTUS_CORE_FRAMEWORK_SESSION_STATE_H_

#include <memory>
#include <mutex>
#include <vector>

#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"

namespace Lotus {
  class OpKernel;
  struct SessionState {
   public:
    SessionState() = default;
    
   SessionState(int num_nodes): session_kernels_(num_nodes) {
     // TODO Dummy constructor for now to add a basic test.
   }
    
    void Init(LotusIR::Graph* graph) {
      p_graph_ = graph;
      session_kernels_.resize(p_graph_->NumberOfNodes());
    }

    // QUESTION: will the indices of the nodes change after the graph is
    // transformed?
    OpKernel* GetKernel(LotusIR::NODEINDEX nodeId);
    void AddKernel(LotusIR::NODEINDEX nodeId, std::unique_ptr<OpKernel> p_kernel);
    
    // state
    // cache of the constructed kernels to avoid spending construction
    // time per executor
    std::vector<unique_ptr<OpKernel>> session_kernels_;
    LotusIR::Graph* p_graph_ = nullptr; // owned by the Model inside an InferenceSession

    // TODO add more
  };
}

#endif // LOTUS_CORE_FRAMEWORK_SESSION_STATE_H_

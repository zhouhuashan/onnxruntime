#ifndef LOTUS_CORE_FRAMEWORK_SESSION_STATE_H_
#define LOTUS_CORE_FRAMEWORK_SESSION_STATE_H_

#include <memory>
#include <mutex>
#include <vector>

#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"

namespace Lotus {
  class SessionState {
 public:
    // TODO constructor
    SessionState() {}
    
 SessionState(int num_nodes): session_kernels_(num_nodes) {
      // TODO Dummy constructor for now to add a basic test.
    }
    
    // how is the session state going to be constructed? with a graph ptr?
    OpKernel* GetKernel(LotusIR::NODEINDEX nodeId);
    void AddKernel(LotusIR::NODEINDEX nodeId, std::unique_ptr<OpKernel> p_kernel);
    
 private:
    // cache of the constructed kernels to avoid spending construction
    // time per executor
    std::vector<unique_ptr<OpKernel>> session_kernels_;
    std::unique_ptr<LotusIR::Graph> p_graph_ = nullptr;
    std::mutex state_lock_;
    // TODO add more
  };
}
  
#endif // LOTUS_CORE_FRAMEWORK_SESSION_STATE_H_

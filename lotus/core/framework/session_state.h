#ifndef LOTUS_CORE_FRAMEWORK_SESSION_STATE_H_
#define LOTUS_CORE_FRAMEWORK_SESSION_STATE_H_

#include <memory>
#include <mutex>
#include <vector>

#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "core/framework/execution_provider.h"

namespace Lotus {
  class SessionState {
   public:
    SessionState() = default;
    
   SessionState(int num_nodes): session_kernels_(num_nodes) {
     // TODO Dummy constructor for now to add a basic test.
   }
    
    void Init(const LotusIR::Graph* graph);
    const LotusIR::Graph* GetGraph() const;

    // kernels
    OpKernel* GetKernel(LotusIR::NODEINDEX node_id) const;
    void AddKernel(LotusIR::NODEINDEX node_id, std::unique_ptr<OpKernel> p_kernel);
    const std::vector<unique_ptr<OpKernel>>& GetKernelVector() const;

    // exec providers
    IExecutionProvider* GetExecutionProvider(const std::string& provider_id) const;
    void AddExecutionProvider(const std::string& provider_id, std::unique_ptr<IExecutionProvider> exec_provider);
    const std::vector<std::unique_ptr<IExecutionProvider>>& GetExecutionProviders() const;

   private:
    // cache of the constructed kernels to avoid spending construction
    // time per executor
    std::vector<unique_ptr<OpKernel>> session_kernels_;
    const LotusIR::Graph* p_graph_ = nullptr; // owned by the Model inside an InferenceSession

    struct ExecutionProviderSet {
      std::vector<std::unique_ptr<IExecutionProvider>> exec_providers;
      std::unordered_map<std::string, size_t> provider_idx_map; // this merely exists to facilitate fast lookup
    };
    ExecutionProviderSet exec_provider_set_;

    // TODO add more
  };
}

#endif // LOTUS_CORE_FRAMEWORK_SESSION_STATE_H_

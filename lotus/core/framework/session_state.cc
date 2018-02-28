#include "core/framework/session_state.h"

namespace Lotus {

OpKernel* SessionState::GetKernel(LotusIR::NODEINDEX nodeId) {
  std::lock_guard<std::mutex> gl(state_lock_);
  if (nodeId >= session_kernels_.size()) {
    return nullptr;
  }
  return session_kernels_[nodeId].get();
}

void SessionState::AddKernel(LotusIR::NODEINDEX nodeId, std::unique_ptr<OpKernel> p_kernel) {
  std::lock_guard<std::mutex> gl(state_lock_);
  // assumes vector is already resize()'ed to the number of nodes in the graph
  // and the nodeIds space is dense
  session_kernels_[nodeId] = std::move(p_kernel);
}

}

#include "core/graph/graph_transformer.h"

namespace LotusIR {

Status GraphTransformerManager::ApplyAll(Graph* graph) {
  bool changed = false;
  for (int step = 0; step < steps_; ++step) {
    for (auto transformer : transformers_) {
      bool t_changed = false;
      Status s = transformer->Apply(graph, &t_changed);
      if (!s.Ok()) return s;
      changed = changed || t_changed;
    }
    if (!changed) break;
  }
  return Status::OK();
}

Status GraphTransformerManager::SetDefault() {
  return Status::OK();
}

}  // namespace LotusIR  

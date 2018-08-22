#pragma once

#include "core/graph/graph_transformer.h"

namespace LotusIR {
// Manages a list of graph transformers. It is initialized with a list of graph
// transformers. Each inference session can further register additional ones.
class GraphTransformerManager {
 public:
  explicit GraphTransformerManager(unsigned steps) noexcept : steps_(steps) {
    // TODO: Register default transformers.
  }

  // Register a graph transformer.
  ::Lotus::Common::Status Register(std::unique_ptr<GraphTransformer> transformer) {
    transformers_.push_back(std::move(transformer));
    return ::Lotus::Common::Status::OK();
  }

  // Apply the list of graph transformers registered on the specified graph
  // up to the given number of steps.
  ::Lotus::Common::Status ApplyAll(Graph& graph) const;

 private:
  GraphTransformerManager() = default;
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(GraphTransformerManager);

  std::vector<std::unique_ptr<GraphTransformer>> transformers_;
  const unsigned steps_;
};
}  // namespace LotusIR

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/graph.h"
#include "core/graph/graph_transformer.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class InsertCastTransformer : public onnxruntime::GraphTransformer {
 public:
  InsertCastTransformer(const std::string& name)
      : onnxruntime::GraphTransformer(name, "Transformer to insert cast node that casts float16 to float for cpu nodes") {
  }

  void AddKernelRegistries(const std::vector<const KernelRegistry*>& kernels) {
    for (auto* kernel : kernels) {
      if (kernel)
        kernels_registries_.push_back(kernel);
    }
  }

  void AddKernelRegistry(const KernelRegistry& kernel) {
    kernels_registries_.push_back(&kernel);
  }

  Status Apply(onnxruntime::Graph& graph, bool& modified) const override;

 private:
  bool NeedInsertCast(const onnxruntime::Node* node, const onnxruntime::NodeArg* input) const;

  std::vector<const KernelRegistry*> kernels_registries_;
};
}  // namespace onnxruntime

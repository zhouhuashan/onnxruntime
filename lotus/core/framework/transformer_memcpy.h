#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

// implements MemCpy node insertion in graph transform
// note that GraphTransformer::Apply() is supposed to be stateless, so this cannot derive from GraphTranformer
class TransformerMemcpyImpl {
 public:
  TransformerMemcpyImpl(LotusIR::Graph& graph, const std::string& provider)
      : graph_(graph), provider_(provider) {}

  bool ModifyGraph(const KernelRegistryManager& schema_registries);

 private:
  void ProcessDefs(LotusIR::Node& node, const KernelRegistryManager& kernel_registries);
  void AddCopyNode(const LotusIR::NodeArg* arg, bool is_input);
  void ProcessInitializers();

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(TransformerMemcpyImpl);

  // use value-based compare to make sure transformer output order is consistent
  struct NodeCompare {
    bool operator()(const LotusIR::Node* lhs, const LotusIR::Node* rhs) const {
      return lhs->Index() < rhs->Index();
    }
  };

  // use value-based compare to make sure transformer output order is consistent
  struct NodeArgCompare {
    bool operator()(const LotusIR::NodeArg* lhs, const LotusIR::NodeArg* rhs) const {
      return lhs->Name() < rhs->Name();
    }
  };

  std::set<LotusIR::Node*, NodeCompare> provider_nodes_;
  std::set<const LotusIR::NodeArg*, NodeArgCompare> non_provider_input_defs_;   // all input defs of non-provider nodes
  std::set<const LotusIR::NodeArg*, NodeArgCompare> non_provider_output_defs_;  // all output defs of non-provider nodes
  std::set<const LotusIR::NodeArg*, NodeArgCompare> provider_input_defs_;       // all input defs of provider nodes that should be in provider allocator
  std::set<const LotusIR::NodeArg*, NodeArgCompare> provider_output_defs_;      // all output defs of provider nodes that should be in provider allocator
  std::map<const LotusIR::NodeArg*, LotusIR::NodeArg*> replacements_;
  LotusIR::Graph& graph_;
  std::string provider_;
};

}  // namespace Lotus

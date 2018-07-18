#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {

// implements MemCpy node insertion in graph transform
// note that GraphTransformer::Apply() is supposed to be stateless, so this cannot derive from GraphTranformer
class TransformerMemcpyImpl {
 public:
  TransformerMemcpyImpl(LotusIR::Graph* graph, const std::string& provider)
      : graph_(graph), provider_(provider) {}

  bool ModifyGraph();

 private:
  void ProcessDefs(LotusIR::Node& node);
  void AddCopyNode(const LotusIR::NodeArg* arg, bool is_input);
  void ProcessInitializers();

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(TransformerMemcpyImpl);

  std::set<LotusIR::Node*> provider_nodes_;
  std::set<const LotusIR::NodeArg*> non_provider_input_defs_;   // all input defs of non-provider nodes
  std::set<const LotusIR::NodeArg*> non_provider_output_defs_;  // all output defs of non-provider nodes
  std::set<const LotusIR::NodeArg*> provider_input_defs_;       // all input defs of provider nodes that should be in provider allocator
  std::set<const LotusIR::NodeArg*> provider_output_defs_;      // all output defs of provider nodes that should be in provider allocator
  std::map<const LotusIR::NodeArg*, LotusIR::NodeArg*> replacements_;
  LotusIR::Graph* graph_;
  std::string provider_;
};

}  // namespace Lotus

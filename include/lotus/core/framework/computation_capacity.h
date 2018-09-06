#pragma once
#include "core/common/common.h"
#include "core/graph/indexed_sub_graph.h"

namespace Lotus {
//A structure to descripe which part of the graph could be run, and how to run it.
struct ComputationCapacity {
  // IndexdSubGraph descript the subgraph that current XP could run, it could be a single node or multiple nodes.
  std::unique_ptr<IndexedSubGraph> sub_graph_;
  // If execution provider want to fuse a sub-graph into a kernel generated on the fly (means can't be pre-defined.)
  // It can set the kernel create function in this field, so Lotus will create the kernel for it.
  // Otherwise Lotus will search kernels in pre-defined kernel registry provided by XP.
  KernelCreateFn fuse_kernel_function_;

  //TODO: if there is a FusedKernelFn attached, Lotus will generate the default KernelDefinition for it, according to the OpSchema generated on the fly
  //A default kernel definition will cover the basic fields, like the input/output constrains.
  //If execution provider want to set some advanced fields on kernel definition, like memory placement / in-place annotation.
  //A KernelDefinitionDecrator function is needed.

  ComputationCapacity() : sub_graph_(nullptr), fuse_kernel_function_(nullptr) {}

  ComputationCapacity(std::unique_ptr<IndexedSubGraph> sub_graph, KernelCreateFn kernel_func)
      : sub_graph_(std::move(sub_graph)),
        fuse_kernel_function_(kernel_func) {}
};
}  // namespace Lotus

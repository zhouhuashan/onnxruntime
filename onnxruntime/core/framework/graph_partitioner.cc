// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/graph_partitioner.h"

#include "core/framework/kernel_registry_manager.h"
#include "core/graph/function.h"
#include "core/graph/graph.h"
#include "core/framework/computation_capacity.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/func_kernel.h"
#include "core/framework/session_state.h"

// uncomment this line to count non-CUDA ops in ONNX domain
//#define COUNT_NON_CUDA_OPS

#ifdef COUNT_NON_CUDA_OPS
class NonCudaOps {
 public:
  ~NonCudaOps() {
    printf("Non-CUDA ops:\n");
    for (auto i : map_) {
      printf("%s: %d\n", i.first.c_str(), i.second);
    }
  }

  void AddOp(const std::string& name) {
    if (map_.count(name))
      map_.at(name)++;
    else
      map_.insert({name, 1});
  }

 private:
  std::map<std::string, int> map_;
};

NonCudaOps non_cuda;
#endif

using namespace ::onnxruntime::common;
namespace onnxruntime {

KernelDefBuilder& BuildFusedKernelDef(KernelDefBuilder& builder, const onnxruntime::Node& node) {
  auto schema = node.Op();
  builder.SetName(schema->Name())
      .SetDomain(schema->domain())
      .SinceVersion(schema->SinceVersion())
      .Provider(node.GetExecutionProviderType());
  auto& inputs = node.InputDefs();
  for (auto input : inputs) {
    builder.TypeConstraint(input->Name(), DataTypeImpl::TypeFromProto(*input->TypeAsProto()));
  }
  return builder;
}

Status GraphPartitioner::Partition(onnxruntime::Graph& graph, const SessionState& session_state) const {
  if (providers_.Empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No provider specified.");
  }
  //fused_kernel_registry is prepareing the kernels created on the fly for fused sub graph.
  //It is only visiable for current session.
  std::shared_ptr<KernelRegistry> fused_kernel_registry = std::make_shared<KernelRegistry>();
  // Partitioning <graph> based on provider preference and their capabilities.
  auto kernel_registries = kernel_registry_mgr_.GetAllKernelRegistries();
  for (auto& provider : providers_) {
    auto capability_results = provider->GetCapability(graph, kernel_registries);
    int count = 0;
    std::vector<Node*> fused_nodes;
    for (auto& capacity : capability_results) {
      if (nullptr == capacity || nullptr == capacity->sub_graph_) {
        continue;
      }
      if (1 == capacity->sub_graph_->nodes.size()) {
        // The <provider> can run a single node in the <graph>.
        auto node = graph.GetNode(capacity->sub_graph_->nodes[0]);
        if (nullptr != node && node->GetExecutionProviderType().empty()) {
          node->SetExecutionProviderType(provider->Type());
        }
      } else {
        // The <provider> can run a fused <sub_graph> in the <graph>.
        //
        // Add fused node into <graph>
        ONNXRUNTIME_ENFORCE(nullptr != capacity->sub_graph_->GetMetaDef());
        std::string node_name = provider->Type() + "_" + capacity->sub_graph_->GetMetaDef()->name + "_" + std::to_string(count++);
        auto fused_node = graph.FuseSubGraph(std::move(capacity->sub_graph_), node_name);
        fused_node->SetExecutionProviderType(provider->Type());

        fused_nodes.push_back(fused_node);
      }
    }
    if (fused_nodes.size() > 0) {
      if (session_state.ExportDll()) {
        std::string dll_path;
        ONNXRUNTIME_RETURN_IF_ERROR(provider->Compile(fused_nodes, dll_path));
        for (auto* node : fused_nodes)
          ONNXRUNTIME_RETURN_IF_ERROR(const_cast<FuseFuncManager*>(session_state.GetFusedFuncMgr())->AddFuncInfo(node->Name(), dll_path));
      } else {
        std::vector<NodeComputeInfo> node_compute_funcs;
        ONNXRUNTIME_RETURN_IF_ERROR(provider->Compile(fused_nodes, node_compute_funcs));
        ONNXRUNTIME_ENFORCE(node_compute_funcs.size() == fused_nodes.size(), "Provider doesn't return correct number of compiled functions");
        for (auto i = 0; i < fused_nodes.size(); i++)
          ONNXRUNTIME_RETURN_IF_ERROR(const_cast<FuseFuncManager*>(session_state.GetFusedFuncMgr())->AddFuncInfo(fused_nodes[i]->Name(), node_compute_funcs[i].compute_func, node_compute_funcs[i].create_state_func, node_compute_funcs[i].release_state_func));
      }
      for (auto* node : fused_nodes) {
        //prepare the func kernel
        KernelDefBuilder builder;
        BuildFusedKernelDef(builder, *node);
        fused_kernel_registry->Register(builder, [](const OpKernelInfo& info) { return new FunctionKernel(info); });
      }
    }
  }

  //For some cases, like fp16 on cpu, right now we don't have any kernel support that.
  //But we will insert cast op to run the model, so skip the error checking here.
  //If after graph transform phase, the node still not assigned, we will report error
  //during kernel creation phase.
#ifdef COUNT_NON_CUDA_OPS
  for (auto& node : graph.Nodes()) {
    if (node.GetExecutionProviderType() != kCudaExecutionProvider &&
        node.Domain() != kMLDomain &&
        node.Domain() != kMSDomain)
      non_cuda.AddOp(node.OpType());
  }
#endif

  kernel_registry_mgr_.RegisterKernelRegistry(fused_kernel_registry, KernelRegistryPriority::HighPriority);

  return Status::OK();
}
}  // namespace onnxruntime

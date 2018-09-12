#include "core/framework/graph_partitioner.h"

#include "core/framework/kernel_registry_manager.h"
#include "core/graph/function.h"
#include "core/graph/graph.h"
#include "core/framework/computation_capacity.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/kernel_registry.h"

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

Status GraphPartitioner::Partition(onnxruntime::Graph& graph) const {
  if (providers_.Empty()) {
    return Status(LOTUS, INVALID_ARGUMENT, "No provider specified.");
  }
  //fused_kernel_registry is prepareing the kernels created on the fly for fused sub graph.
  //It is only visiable for current session.
  std::shared_ptr<KernelRegistry> fused_kernel_registry = std::make_shared<KernelRegistry>();
  // Partitioning <graph> based on provider preference and their capabilities.
  auto kernel_registries = kernel_registry_mgr_.GetAllKernelRegistries();
  for (auto& provider : providers_) {
    auto capability_results = provider->GetCapability(graph, kernel_registries);
    int count = 0;
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
        LOTUS_ENFORCE(nullptr != capacity->sub_graph_->GetMetaDef());
        std::string node_name = provider->Type() + "_" + capacity->sub_graph_->GetMetaDef()->name + "_" + std::to_string(count++);
        auto fused_node = graph.FuseSubGraph(std::move(capacity->sub_graph_), node_name);
        fused_node->SetExecutionProviderType(provider->Type());
        auto fused_kernel_func = capacity->fuse_kernel_function_;
        if (fused_kernel_func != nullptr) {
          // build the kernel definition on the fly, and register it to the fused_kernel_regisitry.
          KernelDefBuilder builder;
          BuildFusedKernelDef(builder, *fused_node);
          fused_kernel_registry->Register(builder, fused_kernel_func);
        }
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

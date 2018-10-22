#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/common/common.h"
#include "core/providers/brainslice/fpga_util.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/computation_capacity.h"
#include "core/providers/brainslice/brainslice_kernel.h"

namespace onnxruntime {
namespace brainslice {
class BrainSlicePinnedAllocator : public CPUAllocator {
 public:
  virtual const ONNXRuntimeAllocatorInfo& Info() const override {
    static ONNXRuntimeAllocatorInfo bs_cpu_allocator_info("BrainSlice",
                                                          ONNXRuntimeAllocatorType::ONNXRuntimeDeviceAllocator, 0,
                                                          ONNXRuntimeMemType::ONNXRuntimeMemTypeCPU);
    return bs_cpu_allocator_info;
  }
};

//In current framework, I have to provide a default allocator, otherwise our allocation planner can't get the default allocator info
//Altough we won't use this default allocator to allocate anything....
//Looks something wrong is the design, may need to take a look later.
class BrainSliceAllocator : public IAllocator {
 public:
  virtual const ONNXRuntimeAllocatorInfo& Info() const override {
    static ONNXRuntimeAllocatorInfo bs_default_allocator_info("BrainSlice",
                                                          ONNXRuntimeAllocatorType::ONNXRuntimeDeviceAllocator, 0,
                                                          ONNXRuntimeMemType::ONNXRuntimeMemTypeDefault);
    return bs_default_allocator_info;
  }

  void* Alloc(size_t size) override {
    ONNXRUNTIME_THROW("BrainSlice has no default allocator");
  }

  void Free(void* p) override {
    ONNXRUNTIME_THROW("BrainSlice has no default allocator");
  }
};

BrainSliceExecutionProvider::BrainSliceExecutionProvider(const fpga::FPGAInfo& info) : handle_(info),
                                                                                       matrix_rf_planner_(std::make_unique<BrainSliceMemoryPlanner>(ISA_Mem_MatrixRf, handle_.GetCapacities().m_bsParameters.MATRIX_RF_SIZE)),
                                                                                       multiply_vrf_planner_(std::make_unique<BrainSliceMemoryPlanner>(ISA_Mem_MultiplyVrf, handle_.GetCapacities().m_bsParameters.MULTIPLY_VRF_SIZE)),
                                                                                       add_sub_vrf_planner_(std::make_unique<BrainSliceMemoryPlanner>(ISA_Mem_AddSubVrf, handle_.GetCapacities().m_bsParameters.ADDSUB_VRF_SIZE)) {
  // insert cpu memory allocator
  AllocatorPtr cpu_allocator(new BrainSlicePinnedAllocator());
  AllocatorPtr bs_allocator(new BrainSliceAllocator());
  InsertAllocator(ONNXRuntimeMemTypeCPU, cpu_allocator);
  InsertAllocator(ONNXRuntimeMemTypeDefault, bs_allocator);
}

Status BrainSliceExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}

bool BrainSliceExecutionProvider::CheckNodeWithCapacity(const onnxruntime::Graph& graph, const onnxruntime::Node& node) const {
  //TODO: right now we only handle GRU node (maybe LSTM later) because they are built-in firmware.
  //Wil need more work to support different node's capacity check.
  if (node.OpType() == "GRU") {
    //1. check batch size is 1
    auto inputs = node.InputDefs();
    if (inputs.size() < 3 || !inputs[0] || !inputs[0]->Shape())
      return false;
    auto x_shape = inputs[0]->Shape();
    if (x_shape->dim_size() != 3 || x_shape->dim()[1].dim_value() != 1)
      return false;
    //2. check W and R is initializer
    auto W = inputs[1];
    auto R = inputs[2];
    const onnx::TensorProto* tmp;
    if (!graph.GetInitializedTensor(W->Name(), tmp) || !graph.GetInitializedTensor(R->Name(), tmp))
      return false;
    //3. check B is we have bias
    if (inputs.size() >= 3 && inputs[3] && !graph.GetInitializedTensor(inputs[3]->Name(), tmp))
      return false;
    auto& attributes = node.GetAttributes();
    //4. bidirection is not supported yet
    auto it = attributes.find("direction");
    if (it != attributes.end() && it->second.s() != "forward")
      return false;
    //5. check activate function
    it = attributes.find("activations");
    if (it != attributes.end()) {
      if (it->second.strings_size() != 2 || (it->second.strings()[0] != "Sigmoid" && it->second.strings()[1] != "Tanh"))
        return false;
    }
    // 6. clip not supported now.
    if (attributes.count("clip") > 0)
      return false;

    // 7. linear_before_reset not supported
    it = attributes.find("linear_before_reset");
    if (it != attributes.end() && it->second.i() != 0)
      return false;
    // TODO: check capacity and the weight size.
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<ComputationCapacity>>
BrainSliceExecutionProvider::GetCapability(const onnxruntime::Graph& graph,
                                           const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputationCapacity>> result;
  for (auto& node : graph.Nodes()) {
    if (graph.IsSourceNode(node) || graph.IsSinkNode(node)) {
      continue;
    }

    if (CheckNodeWithCapacity(graph, node)) {
      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      result.push_back(std::make_unique<ComputationCapacity>(std::move(sub_graph), nullptr));
      //TODO: right now BrainSlice only support one node on the device.
      //Will fix it later.
      break;
    }
  }

  return result;
}

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kBrainSliceExecutionProvider, kOnnxDomain, 7, GRU);

void RegisterBrainSliceKernels(std::function<void(KernelCreateInfo&&)> fn) {
  fn(brainslice::BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kBrainSliceExecutionProvider, kOnnxDomain, 7, GRU)>());
}

std::shared_ptr<KernelRegistry> BrainSliceExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>(onnxruntime::brainslice::RegisterBrainSliceKernels);
  return kernel_registry;
}

BrainSliceMemoryPlanner* BrainSliceExecutionProvider::GetBrainSliceMemoryPlanner(ISA_Mem mem_type) {
  switch (mem_type) {
    case ISA_Mem_AddSubVrf:
      return add_sub_vrf_planner_.get();
    case ISA_Mem_MatrixRf:
      return matrix_rf_planner_.get();
    case ISA_Mem_MultiplyVrf:
      return multiply_vrf_planner_.get();
    default:
      return nullptr;
  }
}
}  // namespace brainslice
}  // namespace onnxruntime

#include "cuda_common.h"
#include "cuda_execution_provider.h"

namespace Lotus {

CUDATransformer::CUDATransformer(const std::string& name)
    : LotusIR::GraphTransformer(name, "Transformer for CUDA execution provider") {
}

Status CUDATransformer::Apply(LotusIR::Graph* graph, bool* modified) const {
  for (auto& node : graph->Nodes()) {
    if (graph->IsSourceNode(node) || graph->IsSinkNode(node))
      continue;

    if (node.GetExecutionProvider().empty()) {
      node.SetExecutionProvider(LotusIR::kCudaExecutionProvider);
      *modified = true;
    }
  }

  return Common::Status::OK();
}

CUDAExecutionProvider::CUDAExecutionProvider(const CUDAExecutionProviderInfo& info)
    : transformer_(info.name), device_id_(info.device_id), cublas_handle_(nullptr) {
  CUDA_CALL_THROW(cudaSetDevice(device_id_));
  CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
  auto& device_factories = DeviceAllocatorRegistry::Instance().AllRegistrations();
  auto cuda_allocator_creator = device_factories.find(CUDA);
  if (cuda_allocator_creator != device_factories.end())
    arena_ = CreateArena(cuda_allocator_creator->second);
}

Status CUDAExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) {
  if (src.Shape().Size() != dst.Shape().Size()) {
    return Status(LOTUS, FAIL, "Tensor size mismatch");
  }

  if (src.Location().name != CUDA && dst.Location().name != CUDA) {
    return Status(LOTUS, FAIL, "Unsupported tensor location");
  }

  size_t bytes = src.DataType()->Size() * src.Shape().Size();
  bool succeeded = false;
  if (dst.Location().name != CUDA) {
    succeeded = CUDA_CALL(cudaMemcpy(dst.MutableData<float>(), src.Data<float>(), bytes, cudaMemcpyDeviceToHost));
  } else if (src.Location().name != CUDA) {
    succeeded = CUDA_CALL(cudaMemcpy(dst.MutableData<float>(), src.Data<float>(), bytes, cudaMemcpyHostToDevice));
  } else {
    succeeded = CUDA_CALL(cudaMemcpy(dst.MutableData<float>(), src.Data<float>(), bytes, cudaMemcpyDeviceToDevice));
  }

  return succeeded ? Status::OK() : Status(LOTUS, FAIL);
}

}  // namespace Lotus

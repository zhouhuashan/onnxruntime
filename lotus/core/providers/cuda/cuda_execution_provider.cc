#include "cuda_common.h"
#include "cuda_execution_provider.h"
#include "core/framework/transformer_memcpy.h"
#include "core/framework/memcpy.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("MemcpyFromHost")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCudaExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
                Memcpy);

REGISTER_KERNEL(KernelDefBuilder("MemcpyToHost")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCudaExecutionProvider)
                    .MemoryType<kMemTypeCPU>(0)
                    .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
                Memcpy);

CUDATransformer::CUDATransformer(const std::string& name)
    : LotusIR::GraphTransformer(name, "Transformer for CUDA execution provider") {
}

Status CUDATransformer::Apply(LotusIR::Graph* graph, bool* modified) const {
  TransformerMemcpyImpl util(graph, LotusIR::kCudaExecutionProvider);
  *modified = util.ModifyGraph();
  return Status::OK();
}

CUDAExecutionProvider::CUDAExecutionProvider(const CUDAExecutionProviderInfo& info)
    : transformer_(info.name), device_id_(info.device_id), cublas_handle_(nullptr) {
  CUDA_CALL_THROW(cudaSetDevice(device_id_));
  CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
  auto& device_factories = DeviceAllocatorRegistry::Instance().AllRegistrations();

  typedef std::pair<std::string, MemType> AllocCreateInfo;
  std::vector<AllocCreateInfo> all_info({{CUDA, kMemTypeDefault},
                                         {CUDA_PINNED, kMemTypeCPU}});
  for (auto pair : all_info) {
    auto iter = device_factories.find(pair.first);
    if (iter != device_factories.end())
      allocators_.insert(std::make_pair(pair.second, CreateAllocator(iter->second, device_id_)));
  }
}

Status CUDAExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  if (src.Shape().Size() != dst.Shape().Size()) {
    return Status(LOTUS, FAIL, "Tensor size mismatch");
  }

  if (src.Location().name != CUDA && dst.Location().name != CUDA) {
    return Status(LOTUS, FAIL, "Unsupported tensor location");
  }

  size_t bytes = src.DataType()->Size() * src.Shape().Size();
  bool succeeded = false;
  if (dst.Location().name != CUDA) {
    /*
    // TODO: add timestamp/event for dst for async copy
    if (dst.Location().name == CUDA_PINNED) {
      succeeded = CUDA_CALL(cudaMemcpyAsync(dst.MutableData<float>(), src.Data<float>(), bytes, cudaMemcpyDeviceToHost));
    } else */
    succeeded = CUDA_CALL(cudaMemcpy(dst.MutableData<float>(), src.Data<float>(), bytes, cudaMemcpyDeviceToHost));
  } else if (src.Location().name != CUDA) {
    if (src.Location().name == CUDA_PINNED)
      // NOTE: when doing async copy from CPU to GPU, following CUDA calls on the same stream would wait
      // until the copy is finished on GPU
      succeeded = CUDA_CALL(cudaMemcpyAsync(dst.MutableData<float>(), src.Data<float>(), bytes, cudaMemcpyHostToDevice));
    else
      succeeded = CUDA_CALL(cudaMemcpy(dst.MutableData<float>(), src.Data<float>(), bytes, cudaMemcpyHostToDevice));
  } else {
    succeeded = CUDA_CALL(cudaMemcpy(dst.MutableData<float>(), src.Data<float>(), bytes, cudaMemcpyDeviceToDevice));
  }

  return succeeded ? Status::OK() : Status(LOTUS, FAIL);
}

}  // namespace Lotus

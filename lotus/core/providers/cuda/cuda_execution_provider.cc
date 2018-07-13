#include "cuda_common.h"
#include "cuda_execution_provider.h"
#include "core/framework/transformer_memcpy.h"
#include "core/framework/memcpy.h"
#include "cuda_fence.h"

namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("MemcpyFromHost")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCudaExecutionProvider)
                    .InputMemoryType<kMemTypeCPUInput>(0)
                    .ExecQueueId(kCudaStreamCopyIn)
                    .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
                Memcpy);

REGISTER_KERNEL(KernelDefBuilder("MemcpyToHost")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCudaExecutionProvider)
                    .OutputMemoryType<kMemTypeCPUOutput>(0)
                    .ExecQueueId(kCudaStreamCopyOut)
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
    : transformer_(info.name), device_id_(info.device_id) {
  CUDA_CALL_THROW(cudaSetDevice(device_id_));
  CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
  CUDNN_CALL_THROW(cudnnCreate(&cudnn_handle_));
  // create streams, default is nullptr
  streams_[kCudaStreamDefault] = nullptr;
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyIn], cudaStreamNonBlocking));
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyOut], cudaStreamNonBlocking));

  auto& device_factories = DeviceAllocatorRegistry::Instance().AllRegistrations();

  typedef std::pair<std::string, MemType> AllocCreateInfo;
  std::vector<AllocCreateInfo> all_info({{CUDA, kMemTypeDefault},
                                         {CUDA_PINNED, kMemTypeCPUOutput}});
  for (auto pair : all_info) {
    auto iter = device_factories.find(pair.first);
    if (iter != device_factories.end())
      allocators_.insert(std::make_pair(pair.second, CreateAllocator(iter->second, device_id_)));
  }
}

CUDAExecutionProvider::~CUDAExecutionProvider() {
  CUDA_CALL_THROW(cudaStreamDestroy(streams_[kCudaStreamCopyIn]));
  CUDA_CALL_THROW(cudaStreamDestroy(streams_[kCudaStreamCopyOut]));
  CUBLAS_CALL_THROW(cublasDestroy(cublas_handle_));
  CUDNN_CALL_THROW(cudnnDestroy(cudnn_handle_));
}

Status CUDAExecutionProvider::Sync() {
  CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  return Status::OK();
}

Status CUDAExecutionProvider::OnRunStart() {
  // check if cudaEvents has passed for deferred release
  auto cpu_alloc = GetAllocator(kMemTypeCPU);
  auto it = deferred_release_cpu_ptr.begin();
  while (it != deferred_release_cpu_ptr.end()) {
    auto& e = it->first;
    auto& v = it->second;
    if (CUDA_CALL(cudaEventQuery(e))) {
      for (auto p : v) {
        cpu_alloc->Free(p);
      }
      it = deferred_release_cpu_ptr.erase(it);
    } else {
      ++it;
    }
  }
  CUDA_RETURN_IF_ERROR(cudaEventCreate(&current_deferred_release_event));
  deferred_release_cpu_ptr.emplace(std::make_pair(current_deferred_release_event, std::vector<void*>()));
  return Status::OK();
}

Status CUDAExecutionProvider::OnRunEnd() {
  // record deferred release event on default stream
  CUDA_RETURN_IF_ERROR(cudaEventRecord(current_deferred_release_event, nullptr));
  return Status::OK();
}

Status CUDAExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  if (src.Shape().Size() != dst.Shape().Size()) {
    return Status(LOTUS, FAIL, "Tensor size mismatch");
  }

  if (src.Location().name != CUDA && dst.Location().name != CUDA) {
    return Status(LOTUS, FAIL, "Unsupported tensor location: src_location is: " + src.Location().name + " and dst_location is: " + dst.Location().name);
  }

  size_t bytes = src.DataType()->Size() * src.Shape().Size();

  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  if (dst.Location().name != CUDA) {
    // copying from CUDA device
    if (dst.Location().name == CUDA_PINNED) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, streams_[kCudaStreamCopyOut]));
    } else
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
  } else if (src.Location().name != CUDA) {
    // copying to CUDA device, the default stream would sync on this fence when using dst as input tensor in kernel
    if (src.Location().name == CUDA_PINNED) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, streams_[kCudaStreamCopyIn]));
    } else
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
  } else {
    // copying between device, use default stream for now
    CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice));
  }

  return Status::OK();
}

}  // namespace Lotus

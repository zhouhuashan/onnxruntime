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

Common::Status CUDAExecutionProvider::Sync() {
  bool status = CUDA_CALL(cudaDeviceSynchronize());
  return status ? Status::OK() : Status(LOTUS, FAIL, "Sync failed.");
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

  bool succeeded = true;
  if (dst.Location().name != CUDA) {
    // copying from CUDA device
    if (dst.Location().name == CUDA_PINNED) {
      succeeded = CUDA_CALL(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, streams_[kCudaStreamCopyOut]));
    } else
      succeeded = CUDA_CALL(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
  } else if (src.Location().name != CUDA) {
    // copying to CUDA device, the default stream would sync on this fence when using dst as input tensor in kernel
    if (src.Location().name == CUDA_PINNED) {
      succeeded = CUDA_CALL(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, streams_[kCudaStreamCopyIn]));
    } else
      succeeded = CUDA_CALL(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
  } else {
    // copying between device, use default stream for now
    succeeded = CUDA_CALL(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice));
  }

  return succeeded ? Status::OK() : Status(LOTUS, FAIL);
}

}  // namespace Lotus

#include "cuda_common.h"
#include "cuda_execution_provider.h"
#include "core/framework/transformer_memcpy.h"
#include "core/framework/memcpy.h"
#include "cuda_fence.h"

using namespace LotusIR;
using namespace Lotus::Common;

namespace Lotus {

namespace Cuda {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder().InputMemoryType<kMemTypeCPUInput>(0).ExecQueueId(kCudaStreamCopyIn)
                      .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder().OutputMemoryType<kMemTypeCPUOutput>(0).ExecQueueId(kCudaStreamCopyOut)
                      .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Memcpy);

} // namespace Cuda

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
      InsertAllocator(pair.second, CreateAllocator(iter->second, device_id_));
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
    if (cudaSuccess == cudaEventQuery(e)) {
      for (auto p : v) {
        cpu_alloc->Free(p);
      }
      it = deferred_release_cpu_ptr.erase(it);
    } else {
      ++it;
    }
  }
  CUDA_RETURN_IF_ERROR(cudaEventCreate(&current_deferred_release_event));
  deferred_release_cpu_ptr.emplace(current_deferred_release_event, std::vector<void*>());
  return Status::OK();
}

Status CUDAExecutionProvider::OnRunEnd() {
  // record deferred release event on default stream
  CUDA_RETURN_IF_ERROR(cudaEventRecord(current_deferred_release_event, nullptr));
  return Status::OK();
}

Status CUDAExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  return CopyTensor(src, dst, kCudaStreamDefault);
}

Status CUDAExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  if (src.Shape().Size() != dst.Shape().Size()) {
    return Status(LOTUS, FAIL, "Tensor size mismatch");
  }

  if (src.Location().name != CUDA && src.Location().name != CUDA_PINNED &&
      dst.Location().name != CUDA && dst.Location().name != CUDA_PINNED) {
    return Status(LOTUS, FAIL, "Unsupported tensor location: src_location is: " + src.Location().name + " and dst_location is: " + dst.Location().name);
  }

  size_t bytes = src.DataType()->Size() * src.Shape().Size();

  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  if (dst.Location().name == CUDA) {
    if (src.Location().name == CUDA_PINNED) {
      // copy from pinned memory to GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, streams_[exec_queue_id]));
    } else if (src.Location().name == CUDA) {
      // copying between GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice));
    } else {
      // copy from other CPU memory to GPU, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
    }
  } else if (src.Location().name == CUDA) {
    if (dst.Location().name == CUDA_PINNED) {
      // copying from GPU to pinned memory, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, streams_[exec_queue_id]));
    } else {
      // copying from GPU to CPU memory, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
    }
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

}  // namespace Lotus

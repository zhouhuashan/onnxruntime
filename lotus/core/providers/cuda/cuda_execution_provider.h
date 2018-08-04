#pragma once
#include "cuda_pch.h"
#include "core/graph/graph_transformer.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "shared_inc/cuda_utils.h"

namespace Lotus {

// Information needed to construct CUDA execution providers.
struct CUDAExecutionProviderInfo {
  std::string name;
  int device_id = 0;
};

enum CUDAStreamType : int {
  kCudaStreamDefault = 0,
  kCudaStreamCopyIn,
  kCudaStreamCopyOut,
  kTotalCudaStreams,
};

class CUDATransformer : public LotusIR::GraphTransformer {
 public:
  CUDATransformer(const std::string& name);
  Status Apply(LotusIR::Graph* graph, bool* modified) const override;
};

// Logical device representation.
class CUDAExecutionProvider : public IExecutionProvider {
 public:
  explicit CUDAExecutionProvider(const CUDAExecutionProviderInfo& info);
  virtual ~CUDAExecutionProvider();

  const LotusIR::GraphTransformer& GetTransformer() const override {
    return transformer_;
  }

  Status Compute(const LotusIR::Node& node, OpKernelContext* /*context*/) const override {
    return Common::Status(
        Common::LOTUS, Common::FAIL,
        "CUDA execution provider: cannot run an op of type `" + node.OpType() + "'.");
  }

  std::string Type() const override {
    return LotusIR::kCudaExecutionProvider;
  }

  Status Sync() override;

  Status OnRunStart() override;

  Status OnRunEnd() override;

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;

  virtual const void* GetExecutionHandle() const noexcept override {
    // The CUDA interface does not return anything interesting.
    return nullptr;
  }

  cublasHandle_t PerThreadCublasHandle() {
    return per_thread_context_->CublasHandle();
  }

  cudnnHandle_t PerThreadCudnnHandle() {
    return per_thread_context_->CudnnHandle();
  }

  cudaStream_t GetStream(int queue_id) const {
    LOTUS_ENFORCE(queue_id >= 0 && queue_id < kTotalCudaStreams);
    return streams_[queue_id];
  }

  const float* GetConstOnes(size_t count) {
    if (!constant_ones_)
      constant_ones_ = Cuda::CreateConstantOnesF();

    return constant_ones_->GetBuffer(count);
  }

  void AddDeferredReleaseCPUPtr(void* p);

  template <typename T>
  inline T* GetScratchBuffer(size_t count_or_bytes) {
    return per_thread_context_->GetScratchBuffer<T>(count_or_bytes);
  }

  void ResetScratchBuffer() {
    return per_thread_context_->ResetScratchBuffer();
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const;

 private:
  cudaStream_t streams_[kTotalCudaStreams];
  std::unique_ptr<Cuda::IConstantBuffer<float>> constant_ones_;
  CUDATransformer transformer_;
  int device_id_;

  struct DeferredReleaseCPUPtrs {
    bool recorded = false;
    std::vector<void*> cpu_ptrs;
  };
  std::unordered_map<cudaEvent_t, DeferredReleaseCPUPtrs> deferred_release_cpu_ptr_;
  std::mutex deferred_release_cpu_ptr_mutex_;

  class PerThreadContext final {
   public:
    PerThreadContext(int device_id, AllocatorPtr gpu_allocator);
    ~PerThreadContext();

    cublasHandle_t CublasHandle() const {
      return cublas_handle_;
    }

    cudnnHandle_t CudnnHandle() const {
      return cudnn_handle_;
    }

    cudaEvent_t& GetCurrentDeferredReleaseEvent() {
      return current_deferred_release_event_;
    }

    void ResetScratchBuffer() {
      gpu_scratch_buffer_bytes_used_ = 0;
    }

    template <typename T>
    inline T* GetScratchBuffer(size_t count_or_bytes) {
      if (count_or_bytes == 0) {
        return nullptr;
      }

      size_t bytes = count_or_bytes;

      if (!std::is_void<T>::value)
        bytes *= sizeof(typename std::conditional<std::is_void<T>::value, void*, T>::type);

      size_t used_bytes = bytes + gpu_scratch_buffer_bytes_used_;
      if (used_bytes > gpu_scratch_buffer_size_) {
        // note that scratch buffer is write-only, so no copy needed when reallocate to a bigger size
        gpu_scratch_buffer_.reset();
        gpu_scratch_buffer_ = IAllocator::MakeUniquePtr<uint8_t>(gpu_allocator_, used_bytes);
        gpu_scratch_buffer_size_ = used_bytes;
      }
      T* p = reinterpret_cast<T*>(gpu_scratch_buffer_.get() + gpu_scratch_buffer_bytes_used_);
      gpu_scratch_buffer_bytes_used_ += bytes;
      return p;
    }

   private:
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;

    // deferred release for temporary CPU pinned memory used in cudaMemcpyAsync
    // note that cudaEvent will be assigned at OnRunEnd() when PerThreadContext destory
    // so the ownership is passed to deferred_release_cpu_ptr_
    cudaEvent_t current_deferred_release_event_ = nullptr;

    // each thread needs to have its own GPU scratch buffer since it
    // needs to be shared within the same thread, while execution in GPU is async.
    // This is different from temporary CPU pinned memory which needs to be hold
    // until GPU copy finished (indicated by deferred_release_event)
    IAllocatorUniquePtr<uint8_t> gpu_scratch_buffer_;
    size_t gpu_scratch_buffer_size_ = 0;
    size_t gpu_scratch_buffer_bytes_used_ = 0;
    AllocatorPtr gpu_allocator_;
  };
  static thread_local std::unique_ptr<PerThreadContext> per_thread_context_;
};

struct KernelCreateInfo;
void RegisterCudaKernels(std::function<void(KernelCreateInfo&&)> create_fn);

}  // namespace Lotus

#pragma once
#include "cuda_pch.h"
#include "core/graph/graph_transformer.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/provider_factories.h"
#include "shared_inc/cuda_utils.h"
#include <deque>

namespace Lotus {

enum CUDAStreamType : int {
  kCudaStreamDefault = 0,
  kCudaStreamCopyIn,
  kCudaStreamCopyOut,
  kTotalCudaStreams,
};

// Logical device representation.
class CUDAExecutionProvider : public IExecutionProvider {
 public:
  explicit CUDAExecutionProvider(const CUDAExecutionProviderInfo& info);
  virtual ~CUDAExecutionProvider();

  virtual AllocatorPtr GetAllocator(MemType mem_type = kMemTypeDefault) const override;

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
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes) const {
    if (count_or_bytes == 0)
      return nullptr;

    return IAllocator::MakeUniquePtr<T>(GetAllocator(kMemTypeDefault), count_or_bytes);
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  cudaStream_t streams_[kTotalCudaStreams];
  std::unique_ptr<Cuda::IConstantBuffer<float>> constant_ones_;
  int device_id_;

  struct DeferredReleaseCPUPtrs {
    bool recorded = false;
    std::vector<void*> cpu_ptrs;
  };
  std::unordered_map<cudaEvent_t, DeferredReleaseCPUPtrs> deferred_release_cpu_ptr_;
  std::mutex deferred_release_cpu_ptr_mutex_;

  class PerThreadContext final {
   public:
    PerThreadContext(int device_id);
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

   private:
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;

    // deferred release for temporary CPU pinned memory used in cudaMemcpyAsync
    // note that cudaEvent will be assigned at OnRunEnd() when PerThreadContext destory
    // so the ownership is passed to deferred_release_cpu_ptr_
    cudaEvent_t current_deferred_release_event_ = nullptr;
  };

  // thread local context during execution
  static thread_local std::shared_ptr<PerThreadContext> per_thread_context_;

  // thread local GPU memory allocator. could be used before execution
  static thread_local AllocatorPtr per_thread_default_allocator_;

  // reuse thread local GPU memory allocator for memory pattern
  mutable std::deque<AllocatorPtr> default_allocator_pool_;
  mutable std::mutex default_allocator_pool_mutex_;

  // reuse thread local context
  mutable std::deque<std::shared_ptr<PerThreadContext>> context_pool_;
  mutable std::mutex context_pool_mutex_;

  void ReleasePerThreadStuffs() const;
};

}  // namespace Lotus

#pragma once
#include "cuda_pch.h"
#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "shared_inc/cuda_call.h"
#include "cuda_execution_provider.h"
#include "shared_inc/fast_divmod.h"

namespace Lotus {
namespace Cuda {

#define CUDA_RETURN_IF_ERROR(expr) LOTUS_RETURN_IF_ERROR(CUDA_CALL(expr) ? Status::OK() : Status(LOTUS, FAIL))
#define CUBLAS_RETURN_IF_ERROR(expr) LOTUS_RETURN_IF_ERROR(CUBLAS_CALL(expr) ? Status::OK() : Status(LOTUS, FAIL))
#define CUSPARSE_RETURN_IF_ERROR(expr) LOTUS_RETURN_IF_ERROR(CUSPARSE_CALL(expr) ? Status::OK() : Status(LOTUS, FAIL))
#define CURAND_RETURN_IF_ERROR(expr) LOTUS_RETURN_IF_ERROR(CURAND_CALL(expr) ? Status::OK() : Status(LOTUS, FAIL))
#define CUDNN_RETURN_IF_ERROR(expr) LOTUS_RETURN_IF_ERROR(CUDNN_CALL(expr) ? Status::OK() : Status(LOTUS, FAIL))
#define CUDNN2_RETURN_IF_ERROR(expr, m) LOTUS_RETURN_IF_ERROR(CUDNN_CALL2(expr, m) ? Status::OK() : Status(LOTUS, FAIL))

// To support cudaMemcpyAsync, the cpu memory should be allocated in pinned memory
// and it can only be released after the copy has finished
template <typename T>
class CudaAsyncBuffer {
 public:
  CudaAsyncBuffer(CUDAExecutionProvider* provider) : provider_(provider), count_(0) {}

  CudaAsyncBuffer(CUDAExecutionProvider* provider, size_t count) : CudaAsyncBuffer(provider) {
    AllocCpuPtr(count);
  }

  CudaAsyncBuffer(CUDAExecutionProvider* provider, const T& value) : CudaAsyncBuffer(provider, 1) {
    *CpuPtr() = value;
  }

  CudaAsyncBuffer(CUDAExecutionProvider* provider, const std::vector<T>& vec) : CudaAsyncBuffer(provider, vec.size()) {
    memcpy(CpuPtr(), vec.data(), vec.size() * sizeof(T));
  }

  void AllocCpuPtr(size_t count) {
    cpu_pinned_copy_ = IAllocator::MakeUniquePtr<T>(provider_->GetAllocator(kMemTypeCPU), count * sizeof(T));
    count_ = count;
  }

  Status CopyToGpu() {
    // note that release gpu_copy_ after launch is OK because it's just going back to arena allocator
    // so the actual GPU operation would have a valid pointer to copy to
    // but CPU memory release needs to be deferred, otherwise if it's reused later from the arena allocator
    // before the copy starts on GPU, the copy would have corrupted data
    gpu_copy_ = IAllocator::MakeUniquePtr<T>(provider_->GetAllocator(), count_ * sizeof(T));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(gpu_copy_.get(), cpu_pinned_copy_.get(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    provider_->AddDeferredReleaseCPUPtr(cpu_pinned_copy_.release());
    return Status::OK();
  }

  T* CpuPtr() const {
    return cpu_pinned_copy_.get();
  }

  gsl::span<T> CpuSpan() const {
    return gsl::span<T>(CpuPtr(), count_);
  }

  T* GpuPtr() const {
    return gpu_copy_.get();
  }

  size_t count() const {
    return count_;
  }

 protected:
  IAllocatorUniquePtr<T> gpu_copy_;
  IAllocatorUniquePtr<T> cpu_pinned_copy_;
  size_t count_;
  CUDAExecutionProvider* provider_;
};

// -----------------------------------------------------------------------
// Base class for CUDA kernels
// -----------------------------------------------------------------------
class CudaKernel : public OpKernel {
 public:
  explicit CudaKernel(const OpKernelInfo& info)
      : OpKernel(info),
        // Is this OK to have a non-const execution provider?
        provider_(const_cast<CUDAExecutionProvider*>(dynamic_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider()))) {
  }

 protected:
  cublasHandle_t CublasHandle() const {
    return provider_->CublasHandle();
  }

  cudnnHandle_t CudnnHandle() const {
    return provider_->CudnnHandle();
  }

  template <typename T>
  inline void AllocateBufferOnGPU(IAllocatorUniquePtr<T>& gpu_copy, size_t count_or_bytes) const {
    auto alloc = provider_->GetAllocator();
    if (count_or_bytes == 0) {
      gpu_copy.release();
      return;
    }

    size_t bytes = count_or_bytes;
    if (!std::is_void<T>::value)
      bytes *= sizeof(typename std::conditional<std::is_void<T>::value, void*, T>::type);

    gpu_copy = IAllocator::MakeUniquePtr<T>(alloc, bytes);
  }

  CUDAExecutionProvider* provider_;
};

// Type mapping for MLFloat16 to half
template <typename T>
class ToCudaType {
 public:
  typedef T MappedType;
};

template <>
class ToCudaType<MLFloat16> {
 public:
  typedef half MappedType;
};

inline bool CalculateFdmStrides(gsl::span<fast_divmod> p, const std::vector<int64_t>& dims) {
  int stride = 1;
  if (p.size() < gsl::narrow_cast<ptrdiff_t>(dims.size()))
    return false;
  auto rank = p.size();
  for (int i = 0; i < rank; i++) {
    p[rank - 1 - i] = fast_divmod(stride);
    if (i < dims.size() - 1) {
      stride *= static_cast<int>(dims[dims.size() - 1 - i]);
    }
  }
  return true;
}

}  // namespace Cuda
}  // namespace Lotus

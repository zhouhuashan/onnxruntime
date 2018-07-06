#pragma once
#include "cuda_pch.h"
#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "shared_inc/cuda_call.h"
#include "cuda_execution_provider.h"

namespace Lotus {
namespace Cuda {

#define CUDA_RETURN_IF_ERROR(expr) LOTUS_RETURN_IF_ERROR(CUDA_CALL(expr) ? Status::OK() : Status(LOTUS, FAIL))
#define CUBLAS_RETURN_IF_ERROR(expr) LOTUS_RETURN_IF_ERROR(CUBLAS_CALL(expr) ? Status::OK() : Status(LOTUS, FAIL))
#define CUSPARSE_RETURN_IF_ERROR(expr) LOTUS_RETURN_IF_ERROR(CUSPARSE_CALL(expr) ? Status::OK() : Status(LOTUS, FAIL))
#define CURAND_RETURN_IF_ERROR(expr) LOTUS_RETURN_IF_ERROR(CURAND_CALL(expr) ? Status::OK() : Status(LOTUS, FAIL))
#define CUDNN_RETURN_IF_ERROR(expr) LOTUS_RETURN_IF_ERROR(CUDNN_CALL(expr) ? Status::OK() : Status(LOTUS, FAIL))
#define CUDNN2_RETURN_IF_ERROR(expr, m) LOTUS_RETURN_IF_ERROR(CUDNN_CALL2(expr, m) ? Status::OK() : Status(LOTUS, FAIL))

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
  inline void AllocateBufferOnGPU(IAllocatorUniquePtr<T>& gpu_copy, size_t count_or_bytes, bool warn_big_buffer) const {
    auto alloc = provider_->GetAllocator();
    if (count_or_bytes == 0) {
      gpu_copy.release();
      return;
    }

    size_t bytes = count_or_bytes;
    if (!std::is_void<T>::value)
      bytes *= sizeof(typename std::conditional<std::is_void<T>::value, void*, T>::type);

    // CUDA inlines cudaMemcpyHostToDevice for size < 64KB, so cudaMemcpy is not blocking GPU
    // Bigger size would synchronize GPU and cpu execution
    if (warn_big_buffer && bytes >= 64 * 1024)
      LOGS_DEFAULT(WARNING) << "CopyToGPU exceeded cudaMemcpyHostToDevice limit and may synchronize CPU/GPU execution";

    gpu_copy = IAllocator::MakeUniquePtr<T>(alloc, bytes);
  }

  template <typename T>
  inline Status CopySmallObjectToGPU(IAllocatorUniquePtr<T>& gpu_copy, const T& cpu_copy) const {
    AllocateBufferOnGPU(gpu_copy, sizeof(T), true);
    CUDA_RETURN_IF_ERROR(cudaMemcpy(gpu_copy.get(), &cpu_copy, sizeof(T), cudaMemcpyHostToDevice));
    return Status::OK();
  }

  template <typename T>
  inline Status CopySmallVectorToGPU(IAllocatorUniquePtr<T>& gpu_vector, const std::vector<T>& cpu_vector) const {
    AllocateBufferOnGPU(gpu_vector, cpu_vector.size(), true);
    CUDA_RETURN_IF_ERROR(cudaMemcpy(gpu_vector.get(), cpu_vector.data(), cpu_vector.size() * sizeof(T), cudaMemcpyHostToDevice));
    return Status::OK();
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

}  // namespace Cuda
}  // namespace Lotus

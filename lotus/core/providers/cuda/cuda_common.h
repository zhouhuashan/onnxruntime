#pragma once
#include "cuda_pch.h"
#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "shared_inc/cuda_call.h"
#include "cuda_execution_provider.h"

namespace Lotus {

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

  template <typename T>
  inline void AllocateSmallBufferOnGPU(IAllocatorUniquePtr<T>& gpu_copy, size_t bytes) const {
    auto alloc = provider_->GetAllocator();
    if (bytes == 0) {
      gpu_copy.release();
      return;
    }

    // CUDA inlines cudaMemcpyHostToDevice for size < 64KB, so cudaMemcpy is not blocking GPU
    // Bigger size would synchronize GPU and cpu execution
    if (bytes >= 64 * 1024)
      LOGS_DEFAULT(WARNING) << "CopyToGPU exceeded cudaMemcpyHostToDevice limit and may synchronize CPU/GPU execution";

    gpu_copy = IAllocator::MakeUniquePtr<T>(alloc, bytes);
  }

  template <typename T>
  inline Status CopySmallObjectToGPU(IAllocatorUniquePtr<T>& gpu_copy, const T& cpu_copy) const {
    AllocateSmallBufferOnGPU(gpu_copy, sizeof(T));
    CUDA_RETURN_IF_ERROR(cudaMemcpy(gpu_copy.get(), &cpu_copy, sizeof(T), cudaMemcpyHostToDevice));
    return Status::OK();
  }

  template <typename T>
  inline Status CopySmallVectorToGPU(IAllocatorUniquePtr<T>& gpu_vector, const std::vector<T>& cpu_vector) const {
    size_t bytes = cpu_vector.size() * sizeof(T);
    AllocateSmallBufferOnGPU(gpu_vector, bytes);
    CUDA_RETURN_IF_ERROR(cudaMemcpy(gpu_vector.get(), cpu_vector.data(), bytes, cudaMemcpyHostToDevice));
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

}  // namespace Lotus

#pragma once
#include "cuda_pch.h"
#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "cuda_call.h"
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
  inline Status CopySmallVectorToGPU(IAllocatorUniquePtr<T>& gpu_vector, std::vector<T> cpu_vector) const {
    auto alloc = provider_->GetAllocator();
    size_t bytes = cpu_vector.size() * sizeof(T);

    // CUDA inlines cudaMemcpyHostToDevice for size < 64KB, so cudaMemcpy is not blocking GPU
    // Bigger size would synchronize GPU and cpu execution
    if (bytes >= 64 * 1024)
      LOGS_DEFAULT(WARNING) << "CopySmallVectorToGPU exceeded cudaMemcpyHostToDevice limit and may synchronize CPU/GPU execution";

    gpu_vector = IAllocator::MakeUniquePtr<T>(alloc, bytes);
    CUDA_RETURN_IF_ERROR(cudaMemcpy(gpu_vector.get(), cpu_vector.data(), bytes, cudaMemcpyHostToDevice));
    return Status::OK();
  }

  CUDAExecutionProvider* provider_;
};

}  // namespace Lotus

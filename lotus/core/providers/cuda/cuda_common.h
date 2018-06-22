#pragma once
#include "cuda_pch.h"
#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "cuda_execution_provider.h"

namespace Lotus {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW>
bool CudaCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg = "");

#define CUDA_CALL(expr) (CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess))
#define CUBLAS_CALL(expr) (CudaCall<cublasStatus_t, false>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS))
#define CUSPARSE_CALL(expr) (CudaCall<cusparseStatus_t, false>((expr), #expr, "CUSPARSE", CUSPARSE_STATUS_SUCCESS))
#define CURAND_CALL(expr) (CudaCall<curandStatus_t, false>((expr), #expr, "CURAND", CURAND_STATUS_SUCCESS))
#define CUDNN_CALL(expr) (CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS))
#define CUDNN_CALL2(expr, m) (CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, m))

#define CUDA_CALL_THROW(expr) (CudaCall<cudaError, true>((expr), #expr, "CUDA", cudaSuccess))
#define CUBLAS_CALL_THROW(expr) (CudaCall<cublasStatus_t, true>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS))
#define CUSPARSE_CALL_THROW(expr) (CudaCall<cusparseStatus_t, true>((expr), #expr, "CUSPARSE", CUSPARSE_STATUS_SUCCESS))
#define CURAND_CALL_THROW(expr) (CudaCall<curandStatus_t, true>((expr), #expr, "CURAND", CURAND_STATUS_SUCCESS))
#define CUDNN_CALL_THROW(expr) (CudaCall<cudnnStatus_t, true>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS))
#define CUDNN_CALL_THROW2(expr, m) (CudaCall<cudnnStatus_t, true>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, m))

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
  cublasHandle_t CublasHandle() const;
  CUDAExecutionProvider* provider_;
};

}  // namespace Lotus

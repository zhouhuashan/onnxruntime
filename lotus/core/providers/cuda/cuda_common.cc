#include "cuda_common.h"
#include "../../common/common.h"

namespace Lotus {

template <typename ERRTYPE>
const char* CudaErrString(ERRTYPE x) {
  LOTUS_NOT_IMPLEMENTED();
}

#define CASE_ENUM_TO_STR(x) \
  case x:                   \
    return #x

template <>
const char* CudaErrString<cudaError_t>(cudaError_t x) {
  cudaDeviceSynchronize();
  return cudaGetErrorString(x);
}
template <>
const char* CudaErrString<cublasStatus_t>(cublasStatus_t e) {
  cudaDeviceSynchronize();

  switch (e) {
    CASE_ENUM_TO_STR(CUBLAS_STATUS_SUCCESS);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_NOT_INITIALIZED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_ALLOC_FAILED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_INVALID_VALUE);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_ARCH_MISMATCH);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_MAPPING_ERROR);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_EXECUTION_FAILED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_INTERNAL_ERROR);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_NOT_SUPPORTED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_LICENSE_ERROR);
    default:
      return "(look for CUBLAS_STATUS_xxx in cublas_api.h)";
  }
}

template <>
const char* CudaErrString<curandStatus>(curandStatus) {
  cudaDeviceSynchronize();
  return "(see curand.h & look for curandStatus or CURAND_STATUS_xxx)";
}

template <typename ERRTYPE, bool THRW>
bool CudaCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg) {
  if (retCode != successCode) {
    try {
#ifdef _WIN32
      auto del = [](char* p) { free(p); };
      std::unique_ptr<char, decltype(del)> hostname_ptr(nullptr, del);
      size_t hostname_len = 0;
      char* hostname = nullptr;
      if (-1 == _dupenv_s(&hostname, &hostname_len, "COMPUTERNAME"))
        hostname = "?";
      else
        hostname_ptr.reset(hostname);
#else
      char hostname[HOST_NAME_MAX];
      if (gethostname(hostname, HOST_NAME_MAX) != 0)
        strcpy(hostname, "?");
#endif
      int currentCudaDevice;
      cudaGetDevice(&currentCudaDevice);
      static char str[1024];
      snprintf(str, 1024, "%s failure %d: %s ; GPU=%d ; hostname=%s ; expr=%s; %s",
               libName, (int)retCode, CudaErrString(retCode), currentCudaDevice,
               hostname,
               exprString, msg);
      if (THRW) {
        LOTUS_THROW(str);
      } else {
        LOGS_DEFAULT(ERROR) << str;
      }
    } catch (const std::exception& e) {  // catch, log, and rethrow since CUDA code sometimes hangs in destruction, so we'd never get to see the error
      if (THRW) {
        LOTUS_THROW(e.what());
      } else {
        LOGS_DEFAULT(ERROR) << e.what();
      }
    }
    return false;
  }
  return true;
}

template bool CudaCall<cudaError, false>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg);
template bool CudaCall<cudaError, true>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg);
template bool CudaCall<cublasStatus_t, false>(cublasStatus_t retCode, const char* exprString, const char* libName, cublasStatus_t successCode, const char* msg);
template bool CudaCall<cublasStatus_t, true>(cublasStatus_t retCode, const char* exprString, const char* libName, cublasStatus_t successCode, const char* msg);

cublasHandle_t CudaKernel::CublasHandle() const {
  return provider_->CublasHandle();
}

}  // namespace Lotus

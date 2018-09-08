#include "test_utils.h"
namespace Lotus {
namespace Test {
IExecutionProvider* TestCPUExecutionProvider() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return &cpu_provider;
}

#ifdef USE_CUDA
IExecutionProvider* TestCudaExecutionProvider() {
  static CUDAExecutionProviderInfo info;
  static CUDAExecutionProvider cuda_provider(info);
  return &cuda_provider;
}
#endif

#ifdef USE_TVM
IExecutionProvider* TestNupharExecutionProvider() {
  static NupharExecutionProviderInfo info;
  static NupharExecutionProvider nuphar_provider(info);
  return &nuphar_provider;
}
#endif  // USE_TVM
}  // namespace Test
}  // namespace Lotus

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "default_providers.h"

#include "core/providers/provider_factories.h"
#ifdef USE_NUPHAR
#include "core/providers/nuphar/provider_factories.h"
#endif

namespace onnxruntime {
namespace test {

std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena) {
  CPUExecutionProviderInfo cpu_pi;
  cpu_pi.create_arena = enable_arena;
  return CreateBasicCPUExecutionProvider(cpu_pi);
}

std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider() {
#ifdef USE_CUDA
  CUDAExecutionProviderInfo cuda_pi;
  return CreateCUDAExecutionProvider(cuda_pi);
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultMkldnnExecutionProvider(bool enable_arena) {
#ifdef USE_MKLDNN
  CPUExecutionProviderInfo mkldnn_pi;
  mkldnn_pi.create_arena = enable_arena;
  return CreateMKLDNNExecutionProvider(mkldnn_pi);
#else
  ONNXRUNTIME_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultNupharExecutionProvider() {
#ifdef USE_NUPHAR
  NupharExecutionProviderInfo nuphar_pi;
  return CreateNupharExecutionProvider(nuphar_pi);
#else
  return nullptr;
#endif
}

}  // namespace test
}  // namespace onnxruntime

#include "core/providers/provider_factories.h"

#include "core/providers/cuda/cuda_execution_provider.h"

namespace onnxruntime {

// Create cuda execution provider
std::unique_ptr<IExecutionProvider>
CreateCUDAExecutionProvider(const CUDAExecutionProviderInfo& info) {
  return std::make_unique<CUDAExecutionProvider>(info);
}

}  // namespace onnxruntime

#include "core/providers/provider_factories.h"

#include "core/providers/cpu/cpu_execution_provider.h"

namespace Lotus {

// Create CPU execution provider
std::unique_ptr<IExecutionProvider>
CreateBasicCPUExecutionProvider(const CPUExecutionProviderInfo &info) {
  return std::make_unique<CPUExecutionProvider>(info);
}

}  // namespace Lotus

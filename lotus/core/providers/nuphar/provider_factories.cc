#include "core/providers/provider_factories.h"
#include "core/providers/nuphar/nuphar_execution_provider.h"

namespace Lotus {

// Create nuphar execution provider
std::unique_ptr<IExecutionProvider>
CreateNupharExecutionProvider(const NupharExecutionProviderInfo& info) {
  return std::make_unique<NupharExecutionProvider>(info);
}

}  // namespace Lotus

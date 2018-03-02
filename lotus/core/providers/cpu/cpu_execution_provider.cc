#include "core/providers/cpu/cpu_execution_provider.h"

namespace Lotus {
    unique_ptr<IExecutionProvider> CreateCPUExecutionProvider(const ExecutionProviderInfo* info)
    {
        return unique_ptr<IExecutionProvider>(new CPUExecutionProvider(info));
    }

    REGISTRY_PROVIDER_CREATOR(CPUExecutionProvider, CreateCPUExecutionProvider);
}
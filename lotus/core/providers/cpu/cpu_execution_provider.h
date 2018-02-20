#ifndef CORE_PROVIDER_CPU_EXECUTION_PROVIDER_H
#define CORE_PROVIDER_CPU_EXECUTION_PROVIDER_H

#include "core/framework/execution_provider.h"

namespace Lotus
{
  // Logical device represenatation.
  class CPUExecutionProvider : public IExecutionProvider
  {
  public:
    CPUExecutionProvider()
    {
    }

  private:
  }

#endif  // CORE_PROVIDER_CPU_EXECUTION_PROVIDER_H

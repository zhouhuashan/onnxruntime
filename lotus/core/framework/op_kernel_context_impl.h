#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"

// Lotus internal OpKernelContext derived class to provide additional
// APIs that aren't desirable to add to the public OpKernelContext API

namespace Lotus {
class SessionState;

class OpKernelContextImpl : public OpKernelContext {
 public:
  explicit OpKernelContextImpl(ExecutionFrame& frame,
                               const OpKernel& kernel,
                               const Logging::Logger& logger)
      : OpKernelContext(&frame, &kernel, logger) {
  }

  const SessionState* SubgraphSessionState(const std::string& attribute_name) {
    return GetSessionState().GetSubgraphSessionState(GetNodeIndex(), attribute_name);
  }
};

}  // namespace Lotus

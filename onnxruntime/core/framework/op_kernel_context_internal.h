// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"

// onnxruntime internal OpKernelContext derived class to provide additional
// APIs that aren't desirable to add to the public OpKernelContext API

namespace onnxruntime {
class SessionState;

class OpKernelContextInternal : public OpKernelContext {
 public:
  explicit OpKernelContextInternal(ExecutionFrame& frame,
                                   const OpKernel& kernel,
                                   const logging::Logger& logger)
      : OpKernelContext(&frame, &kernel, logger) {
  }

  const SessionState* SubgraphSessionState(const std::string& attribute_name) {
    return GetSessionState().GetSubgraphSessionState(GetNodeIndex(), attribute_name);
  }

  const MLValue* GetInputMLValue(int index) const {
    return OpKernelContext::GetInputMLValue(index);
  }

  MLValue* GetOutputMLValue(int index) {
    return OpKernelContext::GetOutputMLValue(index);
  }
};

}  // namespace onnxruntime

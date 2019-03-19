// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include "core/session/session.h"
#include "core/session/inference_session.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

std::unique_ptr<Session> Session::Create(const SessionOptions& session_options,
                                         logging::LoggingManager* logging_manager,
                                         SessionType session_type) {
  switch (session_type) {
    case onnxruntime::Session::SessionType::Inference:
      return std::unique_ptr<InferenceSession>(new InferenceSession(session_options, logging_manager));
      break;
    default:
      break;
  }
  return nullptr;
}
}  // namespace onnxruntime

#ifndef CORE_FRAMEWORK_INFERENCE_SESSION_H
#define CORE_FRAMEWORK_INFERENCE_SESSION_H

#include <vector>
#include <string>

#include "core/framework/execution_provider.h"
#include "core/framework/ml_value.h"
#include "core/common/status.h"

namespace Lotus
{
  struct SessionOptions {
    int num_threads;
    // TODO add more
  };

  // Per model, handling multiple requests.
  class InferenceSession {
  public:
    explicit InferenceSession(const SessionOptions& session_options);

    // Load an ONNX model and initialize.
    Common::Status Load(const std::string& model_name);

    // Both feeds and fetches are owned by client code, and can't be changed
    // by client code during Run().
    Common::Status Run(const std::vector<MLValue>& feeds, std::vector<MLValue>* p_fetches);

    // The list of execution providers in preference order.
    Common::Status SetProviderPreference(const std::vector<IExecutionProvider>& providers);

    ~InferenceSession() = default;

    class Impl;

 private:
    std::unique_ptr<Impl> impl_;
  };
}

#endif  // CORE_FRAMEWORK_INFERENCE_SESSION_H

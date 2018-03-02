#ifndef CORE_FRAMEWORK_INFERENCE_SESSION_H
#define CORE_FRAMEWORK_INFERENCE_SESSION_H

#include "core/framework/execution_provider.h"
#include "core/framework/ml_value.h"
#include "core/common/status.h"
#include "core/common/common.h"
#include "core/framework/executor.h"

namespace Lotus
{
  struct SessionOptions {
    int num_threads;
    vector<ExecutionProviderInfo> ep_infors;
    // TODO add more
  };

  // Per model, handling multiple requests.
  class InferenceSession {
  public:
    explicit InferenceSession(const SessionOptions& session_options);

    ~InferenceSession() = default;

    // Load an ONNX model and initialize.
    Common::Status Load(const std::string& model_name);

    // Both feeds and fetches are owned by client code, and can't be changed
    // by client code during Run().
    Status Run(const std::vector<MLValue>& feeds, std::vector<MLValue>* p_fetches);
    
  private:
      
    class Impl;

 private:
    std::unique_ptr<Impl> impl_;
  };
}

#endif  // CORE_FRAMEWORK_INFERENCE_SESSION_H

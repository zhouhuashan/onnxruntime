#ifndef CORE_FRAMEWORK_INFERENCE_SESSION_H
#define CORE_FRAMEWORK_INFERENCE_SESSION_H

#include <vector>
#include "core/framework/execution_provider.h"
#include "core/framework/ml_value.h"
#include "core/graph/status.h"

using namespace Lotus::Common;

namespace Lotus
{
  // Per model, handling multiple requests.
  class InferenceSession {
  public:
    struct SessionState {

    };

    // Load an ONNX model and initialize.
    Status Load(const std::string& model_name);

    // Both feeds and fetches are owned by client code, and can't be changed
    // by client code during Run().
    Status Run(const std::vector<MLValue>& feeds, std::vector<MLValue>* p_fetches);

    // The list of execution providers in preference order.
    Status SetProviderPreference(const std::vector<IExecutionProvider>& providers);

  private:

    // The model served by this inference session instance.
    std::shared_ptr<Model> m_model;

    // The list of execution providers in preference order.
    std::vector<IExecutionProvider> m_executionProviders;

    // A set of executors that can run in parallel.
    std::vector<Executor> m_executors;

    // The immutable state for each op in the model. Shared by all executors.
    SessionState m_session_state;
  };
}

#endif  // CORE_FRAMEWORK_INFERENCE_SESSION_H

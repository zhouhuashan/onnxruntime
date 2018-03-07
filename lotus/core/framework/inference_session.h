#ifndef LOTUS_CORE_FRAMEWORK_INFERENCE_SESSION_H_
#define LOTUS_CORE_FRAMEWORK_INFERENCE_SESSION_H_

#include "core/framework/execution_provider.h"
#include "core/framework/ml_value.h"
#include "core/common/status.h"
#include "core/common/common.h"
#include "core/platform/types.h"

namespace Lotus
{
  struct SessionOptions {
    int num_threads;
    vector<ExecutionProviderInfo> ep_infors;
    // TODO add more
  };

  struct RunOptions {
    bool enable_debug_mode = false;
    bool enable_sequential_execution = true;
    int64 timeout_in_ms = 0; // TODO choose a good default
    std::string run_tag = ""; // custom tag to tag all the runs
  };

  // Per model, handling multiple requests.
  class InferenceSession {
  public:
    explicit InferenceSession(const SessionOptions& session_options);
    ~InferenceSession();

    // Load an ONNX model and initialize.
    Common::Status Load(const std::string& model_uri);

    Common::Status Initialize();

    // Both feeds and fetches are owned by client code, and can't be changed
    // by client code during Run().
    Common::Status Run(const std::vector<MLValue>& feeds, std::vector<MLValue>* p_fetches);

    Common::Status Run(const RunOptions& run_options,
                       const std::vector<MLValue>& feeds,
                       std::vector<MLValue>* p_fetches);

    // The list of execution providers in preference order.
    Common::Status SetProviderPreference(const std::vector<IExecutionProvider>& providers);

    // Get current num threads running Run
    int GetCurrentNumRuns();

 private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(InferenceSession);
  };
}

#endif  // LOTUS_CORE_FRAMEWORK_INFERENCE_SESSION_H_

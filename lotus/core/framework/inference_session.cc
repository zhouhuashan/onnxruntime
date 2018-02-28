#include "core/framework/inference_session.h"

#include <string>
#include <memory>
#include <mutex>

#include "core/framework/op_kernel.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/lib/threadpool.h"
#include "core/framework/executor.h"
#include "core/framework/session_state.h"

namespace Lotus {

class InferenceSession::Impl {
 public:
  Impl(const SessionOptions& session_options): env_(Env::Default()) {
    const int num_threads = session_options.num_threads;
    // per session threadpool; we can also use the global threadpool instead if required
    thread_pool_.reset(new thread::ThreadPool(env_, "Compute", num_threads));
  }

  // TODO add the methods of the parent class

 private:
  // The model served by this inference session instance.
  std::shared_ptr<Model> model_;
  
  // The list of execution providers in preference order.
  std::vector<IExecutionProvider> execution_providers_;

  // A set of executors that can run in parallel.
  std::vector<Executor> executors_;

  // State for each op in the model. Shared by all executors.
  SessionState session_state_;

  // Environment for this session
  Env* env_;

  // Threadpool for this session
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

InferenceSession::InferenceSession(const SessionOptions& session_options):
    impl_(new Impl(session_options)) {
  
}

Common::Status InferenceSession::Load(const std::string& model_uri) {
  // TODO
  LOTUS_UNUSED_VARIABLE(model_uri);
  return Common::Status::OK();
}

Common::Status InferenceSession::Run(const std::vector<MLValue>& feeds, std::vector<MLValue>* p_fetches) {
  // TODO
  LOTUS_UNUSED_VARIABLE(feeds);
  LOTUS_UNUSED_VARIABLE(p_fetches);
  return Common::Status::OK();
}

Common::Status InferenceSession::SetProviderPreference(const std::vector<IExecutionProvider>& providers) {
  LOTUS_UNUSED_VARIABLE(providers);
  // TODO
  return Common::Status::OK();
}


}

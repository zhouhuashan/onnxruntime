#include "core/framework/inference_session.h"
#include "core/common/logging.h"
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
    auto& provider_mgr = ExecutionProviderMgr::Instance();
    for (auto& info : session_options.ep_infors)
    {
        auto provider = provider_mgr.GetProvider(info.Name(), info);
        if (provider == nullptr)
        {
            LOG(WARNING) << "Execution Provider with name: "
                << info.Name() << "Not found.";
            continue;
        }
        execution_providers_.push_back(std::move(provider));
    }
  }

  // TODO add the methods of the parent class

 private:
  // The model served by this inference session instance.
  std::shared_ptr<Model> model_;
  
  // The list of execution providers in preference order.
  std::vector<unique_ptr<IExecutionProvider> > execution_providers_;

  // A set of executors that can run in parallel.
  std::vector<Executor> executors_;

  // State for each op in the model. Shared by all executors.
  SessionState session_state_;

  // Environment for this session
  // TODO: Should we use naked pointer here?
  // If yes, explain the ownership and lifetime
  Env* env_;

  // Threadpool for this session
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

InferenceSession::InferenceSession(const SessionOptions& session_options):
    impl_(new Impl(session_options)) {
  
}

Status InferenceSession::Load(const std::string& model_uri) {
  // TODO
  UNUSED_PARAMETER(model_uri);
  return Status::OK();
}

Status InferenceSession::Run(const std::vector<MLValue>& feeds, std::vector<MLValue>* p_fetches) {
  // TODO
  UNUSED_PARAMETER(feeds);
  UNUSED_PARAMETER(p_fetches);
  return Status::OK();
}

}

#include "core/framework/inference_session.h"

#include <mutex>

#include "core/common/logging.h"
#include "core/framework/executor.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/lib/threadpool.h"
#include "core/platform/notification.h"

namespace Lotus {

class InferenceSession::Impl {
 public:
  Impl(const SessionOptions& session_options)
      : env_(Env::Default()),
        thread_pool_(env_, "Compute", session_options.num_threads) {
    // QUESTION: what if the user doesn't provide his preferred list of execution
    // providers? Should we have our own default?
    auto& provider_mgr = ExecutionProviderMgr::Instance();
    for (auto& info : session_options.ep_options) {
      auto provider = provider_mgr.GetProvider(info.provider_type, info.provider_info);
      if (provider == nullptr) {
        LOG(WARNING) << "Execution Provider with name: "
                     << info.provider_type << "Not found.";
        continue;
      }
      execution_providers_.insert(std::make_pair(info.provider_type, std::move(provider)));
    }
  }

  // TODO add the methods of the parent class

  Common::Status Load(const std::string& model_uri) {
    std::lock_guard<std::mutex> l(session_mutex_);
    if (is_model_loaded_) {  // already loaded
      LOG(INFO) << "Model: " << model_uri << " has already been loaded.";
      return Common::Status::OK();
    }
    std::shared_ptr<Model> tmp_model_ptr;
    Common::Status st = Model::Load(model_uri, &tmp_model_ptr);
    if (st.IsOK()) {
      is_model_loaded_ = true;
      model_ = tmp_model_ptr;
      session_state_.Init(model_->MainGraph());
    }

    return st;
  }

  Common::Status Initialize() {
    std::lock_guard<std::mutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOG(ERROR) << "Model was not loaded";
      return Common::Status(Common::LOTUS, Common::FAIL, "Model was not loaded.");
    }

    if (is_inited_) {  // already initialized
      LOG(INFO) << "Session has already been initialized.";
      return Common::Status::OK();
    }

    LOTUS_RETURN_IF_ERROR(TransformGraph());
    LOTUS_RETURN_IF_ERROR(ConstructKernels());

    // TODO add other per session initialization stuff here

    is_inited_ = true;
    return Status::OK();
  }

  int GetCurrentNumRuns() {
    return current_num_runs_.load();
  }

  Common::Status Run(const std::vector<MLValue>& feeds, std::vector<MLValue>* p_fetches) {
    RunOptions run_options;
    return Run(run_options, feeds, p_fetches);
  }

  Common::Status Run(const RunOptions& run_options,
                     const std::vector<MLValue>& feeds,
                     std::vector<MLValue>* p_fetches) {
    {
      std::lock_guard<std::mutex> l(session_mutex_);
      if (!is_inited_) {
        LOG(ERROR) << "Session was not initialized";
        return Common::Status(Common::LOTUS, Common::FAIL, "Session not initialized.");
      }
    }

    // TODO add instrumentation to measure the time taken for this Run
    LOG(INFO) << "Running with tag: " << run_options.run_tag << std::endl;
    ++current_num_runs_;

    // TODO should we add this exec to the list of executors? i guess its not needed now?

    struct RunStatus {
      std::unique_ptr<Executor> p_exec;
      Common::Status status;  // used to collect the status from the executor
      Notification executor_done;
    };

    RunStatus run_status;
    if (run_options.enable_sequential_execution) {
      run_status.p_exec = std::move(Executor::NewSequentialExecutor(session_state_));
    } else {
      run_status.p_exec = std::move(Executor::NewParallelExecutor(session_state_));
    }

    thread_pool_.Schedule([&run_options, &feeds, p_fetches, &run_status]() {
      Common::Status local_status = run_status.p_exec->Execute(run_options, feeds, p_fetches);
      run_status.status = local_status;
      run_status.executor_done.Notify();
    });

    // this is a blocking Run, hence wait to be notified by the above closure when the executor is done
    Common::Status waitStatus = WaitForNotification(&run_status.executor_done, run_options.timeout_in_ms);
    Common::Status retval;

    if (!waitStatus.IsOK()) {
      // TODO should we cancel the thread in the pool that corresponds to this executor?
      retval = waitStatus;
    } else {
      retval = run_status.status;
    }

    --current_num_runs_;
    return retval;
  }

 private:
  Common::Status TransformGraph() {
    for (auto& ep : execution_providers_) {
      bool is_modified;
      ep.second->GetTransformer().Apply(*session_state_.p_graph_, is_modified);
    }
    return Status::OK();
  }

  Common::Status ConstructKernels() {
    Graph* graph = session_state_.p_graph_;
    for (auto node_it = graph->Nodes_begin(); node_it != graph->Nodes_end(); ++node_it) {
      std::unique_ptr<OpKernel> op_kernel_ptr = CreateOpKernel(**node_it);
      if (!op_kernel_ptr) {
        LOG(ERROR) << "Could not create kernel for opId: " << (*node_it)->OpType();
        continue;  // TODO for now ignore the error and continue until the actual kernels are ready
        // return Common::Status(Common::LOTUS,
        //                       Common::FAIL,
        //                       "Failed to initialize session because kernel creation failed");
      }
      session_state_.AddKernel((*node_it)->Index(), std::move(op_kernel_ptr));
    }

    return Status::OK();
  }

  std::unique_ptr<OpKernel> CreateOpKernel(const Node& node) {
    const std::string& exec_provider_name = node.Device();
    if (exec_provider_name.empty() || execution_providers_.end() == execution_providers_.find(exec_provider_name)) {
        LOG(ERROR) << "Could not create kernel for node: " << node.Name() << " as there's no execution provider allocated.";
        return nullptr;
    }
    auto& allocator_info = execution_providers_[exec_provider_name]->GetTempSpaceAllocator().Info();
    OpKernelInfo op_kernel_info{node, allocator_info};
    OpKernel* result;
    auto status = KernelRegistry::Instance()->CreateKernel(*(node.Op()), exec_provider_name, op_kernel_info, &result);
    return std::unique_ptr<OpKernel>(result);
  }

  Common::Status WaitForNotification(Notification* p_executor_done, int64 timeout_in_ms) {
    if (timeout_in_ms > 0) {
      LOTUS_NOT_IMPLEMENTED;  // TODO
    } else {
      p_executor_done->WaitForNotification();
    }
    return Status::OK();
  }

  // The model served by this inference session instance.
  std::shared_ptr<Model> model_;

  // The list of execution providers in preference order.
  std::map<std::string, ExecutionProviderPtr> execution_providers_;

  // A set of executors that can run in parallel.
  std::vector<std::unique_ptr<Executor>> executors_;  // TODO do we need this vector?

  // State for each op in the model. Shared by all executors.
  SessionState session_state_;

  // Environment for this session
  Env* env_;  // statically allocated pointer, no need to manage its lifetime.

  // Threadpool for this session
  thread::ThreadPool thread_pool_;

  // Number of concurrently running executors
  std::atomic<int> current_num_runs_;

  std::mutex session_mutex_;      // to ensure only one thread can invoke Load/Initialize
  bool is_model_loaded_ = false;  // GUARDED_BY(session_mutex_)
  bool is_inited_ = false;        // GUARDED_BY(session_mutex_)
};

//
// InferenceSession
//
InferenceSession::InferenceSession(const SessionOptions& session_options) : impl_(new Impl(session_options)) {
}

InferenceSession::~InferenceSession() = default;

Common::Status InferenceSession::Load(const std::string& model_uri) {
  return impl_->Load(model_uri);
}

Common::Status InferenceSession::Initialize() {
  return impl_->Initialize();
}

Common::Status InferenceSession::Run(const std::vector<MLValue>& feeds, std::vector<MLValue>* p_fetches) {
  return impl_->Run(feeds, p_fetches);
}

Common::Status InferenceSession::Run(const RunOptions& run_options,
                                     const std::vector<MLValue>& feeds,
                                     std::vector<MLValue>* p_fetches) {
  return impl_->Run(run_options, feeds, p_fetches);
}

int InferenceSession::GetCurrentNumRuns() {
  return impl_->GetCurrentNumRuns();
}

}  // namespace Lotus

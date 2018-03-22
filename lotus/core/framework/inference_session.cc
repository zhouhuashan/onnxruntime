#include "core/framework/inference_session.h"

#include <mutex>
#include <sstream>

#include "core/common/logging.h"
#include "core/framework/executor.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
//#include "core/platform/env.h"
//#include "core/lib/threadpool.h"
#include "core/framework/execution_frame.h"
#include "core/graph/tensorutils.h"
#include "core/platform/notification.h"

namespace Lotus {

class InferenceSession::Impl {
 public:
  Impl(const SessionOptions& session_options)
      : session_options_(session_options) {
    //env_(Env::Default()) {
    //thread_pool_(env_, "Compute", session_options.num_threads) {
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
      session_state_.AddExecutionProvider(info.provider_type, std::move(provider));
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
    Common::Status status = Model::Load(model_uri, &tmp_model_ptr);
    if (status.IsOK()) {
      is_model_loaded_ = true;
      model_ = tmp_model_ptr;
    }

    return status;
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

    Graph* graph = model_->MainGraph();
    LOTUS_RETURN_IF_ERROR(TransformGraph(*graph));
    LOTUS_RETURN_IF_ERROR(graph->Resolve());

    // at this point the graph should be in a frozen state
    // hence we set it in the session for use by the executors
    session_state_.SetGraph(graph);

    // TODO This function doesn't modify the graph, but I've to still
    // pass it by non-const ref because the graph API doesn't support any
    // const iterator today
    LOTUS_RETURN_IF_ERROR(SaveKernelsAndMLValueNameIndexMapping(*graph));

    // TODO add other per session initialization stuff here

    // get execution plan
    if (session_options_.enable_sequential_execution) {
      // Why use a unique_ptr here? the only other ways to avoid using a unique_ptr are
      // (1) making a copy or (2) passing a ptr to the private session_state var (p_seq_exec_plan) to CreatePlan.
      // Passing a pointer to a private member variable doesn't seem the right thing to do.
      std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan = std::make_unique<SequentialExecutionPlan>();
      // TODO change SimpleAllocationPlanner to use SequentialPlanner; Simple exists for testing only.
      LOTUS_RETURN_IF_ERROR(SimpleAllocationPlanner::CreatePlan(session_state_, p_seq_exec_plan.get()));
      session_state_.SetExecutionPlan(std::move(p_seq_exec_plan));
    } else {
      LOTUS_NOT_IMPLEMENTED;
    }

    is_inited_ = true;
    return Status::OK();
  }

  int GetCurrentNumRuns() {
    return current_num_runs_.load();
  }

  Common::Status Run(const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches) {
    RunOptions run_options;
    return Run(run_options, feeds, output_names, p_fetches);
  }

  Common::Status Run(const RunOptions& run_options,
                     const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches) {
    {
      std::lock_guard<std::mutex> l(session_mutex_);
      if (!is_inited_) {
        LOG(ERROR) << "Session was not initialized";
        return Common::Status(Common::LOTUS, Common::FAIL, "Session not initialized.");
      }
    }

    // TODO add instrumentation to measure the time taken for this Run
    if (!run_options.run_tag.empty()) {
      LOG(INFO) << "Running with tag: " << run_options.run_tag;
    }

    ++current_num_runs_;

    // TODO should we add this exec to the list of executors? i guess its not needed now?

    std::unique_ptr<Executor> p_exec;
    if (session_options_.enable_sequential_execution) {
      p_exec = std::move(Executor::NewSequentialExecutor(session_state_, feeds, output_names));
    } else {
      LOTUS_NOT_IMPLEMENTED;
    }
    Common::Status retval = p_exec->Execute(run_options, feeds, output_names, p_fetches);
    --current_num_runs_;
    return retval;
  }

 private:
  Common::Status TransformGraph(Graph& graph) {
    for (auto& ep : session_state_.GetExecutionProviders()) {
      bool is_modified;
      ep->GetTransformer().Apply(graph, is_modified);
    }
    return Common::Status::OK();
  }

  // This function does the following:
  // - constructs the kernels and saves them in the session state
  // - builds the MLValue name->idx mapping and saves it in the session state
  // The reason we're doing 2 operations in the same function is so that we iterate
  // through all the nodes only once.
  Common::Status SaveKernelsAndMLValueNameIndexMapping(Graph& graph) {
    int curr_idx = 0;
    for (auto node_it = graph.Nodes_begin(); node_it != graph.Nodes_end(); ++node_it) {
      Node* p_node = *node_it;

      // ignore source and sink nodes
      if (graph.IsSourceNode(p_node->Index()) || graph.IsSinkNode(p_node->Index())) {
        continue;
      }

      // construct and save the kernels
      std::unique_ptr<OpKernel> p_op_kernel;
      LOTUS_RETURN_IF_ERROR(CreateOpKernel(*p_node, &p_op_kernel));
      session_state_.AddKernel(p_node->Index(), std::move(p_op_kernel));

      // build the MLValue->index map
      auto& inputs = p_node->InputDefs();
      for (auto& def : inputs) {
        session_state_.AddMLValueNameIdx(def->Name(), curr_idx++);
      }
      auto& outputs = p_node->OutputDefs();
      for (auto def : outputs) {
        session_state_.AddMLValueNameIdx(def->Name(), curr_idx++);
      }
    }

    return Status::OK();
  }

  Common::Status CreateOpKernel(const Node& node, std::unique_ptr<OpKernel>* p_op_kernel) {
    // TODO below line exists to make unit tests work until we've partitioning ready; replace with node.Device()
    const std::string& exec_provider_name = "CPUExecutionProvider";
    if (exec_provider_name.empty() || !session_state_.GetExecutionProvider(exec_provider_name)) {
      std::ostringstream error_msg;
      error_msg << "Could not create kernel for node: " << node.Name() << " as there's no execution provider allocated.";
      LOG(ERROR) << error_msg.str();
      return Common::Status(Common::LOTUS, Common::FAIL, error_msg.str());
    }
    auto& allocator_info = session_state_.GetExecutionProvider(exec_provider_name)->GetTempSpaceAllocator().Info();
    auto status = KernelRegistry::Instance()->CreateKernel(*(node.Op()), exec_provider_name, node, allocator_info, p_op_kernel);
    return status;
  }

  Common::Status WaitForNotification(Notification* p_executor_done, int64 timeout_in_ms) {
    if (timeout_in_ms > 0) {
      LOTUS_NOT_IMPLEMENTED;  // TODO
    } else {
      p_executor_done->WaitForNotification();
    }
    return Status::OK();
  }

  const SessionOptions& session_options_;

  // The model served by this inference session instance.
  // Currently this has to be a shared ptr because the Model::Load method
  // returns a shared_ptr only. Ideally factory functions should always return
  // unique_ptr for maximum flexibility. Client can always upgrade it to shared_ptr
  // if they need.
  std::shared_ptr<Model> model_;

  // A set of executors that can run in parallel.
  std::vector<std::unique_ptr<Executor>> executors_;  // TODO do we need this vector?

  // Immutable state for each op in the model. Shared by all executors.
  SessionState session_state_;

  // Environment for this session
  // not used now; we'll need it when we introduce threadpool
  // statically allocated pointer, no need to manage its lifetime.
  //Env* env_;

  // Threadpool for this session
  //thread::ThreadPool thread_pool_; // not used for now; will add it later when implementing RunAsync

  // Number of concurrently running executors
  std::atomic<int> current_num_runs_;

  std::mutex session_mutex_;      // to ensure only one thread can invoke Load/Initialize
  bool is_model_loaded_ = false;  // GUARDED_BY(session_mutex_)
  bool is_inited_ = false;        // GUARDED_BY(session_mutex_)
};

//
// InferenceSession
//
InferenceSession::InferenceSession(const SessionOptions& session_options) : impl_(std::make_unique<Impl>(session_options)) {
}

InferenceSession::~InferenceSession() = default;

Common::Status InferenceSession::Load(const std::string& model_uri) {
  return impl_->Load(model_uri);
}

Common::Status InferenceSession::Initialize() {
  return impl_->Initialize();
}

Common::Status InferenceSession::Run(const NameMLValMap& feeds,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  return impl_->Run(feeds, output_names, p_fetches);
}

Common::Status InferenceSession::Run(const RunOptions& run_options,
                                     const NameMLValMap& feeds,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  return impl_->Run(run_options, feeds, output_names, p_fetches);
}

int InferenceSession::GetCurrentNumRuns() {
  return impl_->GetCurrentNumRuns();
}

}  // namespace Lotus

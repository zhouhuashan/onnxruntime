#include "core/framework/inference_session.h"

#include <mutex>
#include <sstream>

#include "core/common/logging/logging.h"
#include "core/framework/executor.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
//#include "core/platform/env.h"
//#include "core/lib/threadpool.h"
#include "core/framework/execution_frame.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/tensorutils.h"
#include "core/platform/notification.h"

namespace Lotus {
class InferenceSession::Impl {
 public:
  Impl(const SessionOptions& session_options, Logging::LoggingManager* logging_manager)
      : session_options_{session_options}, logging_manager_{logging_manager} {
    InitLogger(logging_manager);

    //env_(Env::Default()) {
    //thread_pool_(env_, "Compute", session_options.num_threads) {
    // QUESTION: what if the user doesn't provide his preferred list of execution
    // providers? Should we have our own default?
    auto& provider_mgr = ExecutionProviderMgr::Instance();

    VLOGS(*session_logger_, 1) << "Adding execution providers.";
    for (auto& info : session_options.ep_options) {
      auto provider = provider_mgr.GetProvider(info.provider_type, info.provider_info);
      if (provider == nullptr) {
        LOGS(*session_logger_, WARNING) << "Execution Provider with name: " << info.provider_type << "Not found.";
        continue;
      }

      VLOGS(*session_logger_, 1) << "Adding execution provider with name: " << info.provider_type;
      session_state_.AddExecutionProvider(info.provider_type, std::move(provider));
    }
  }

  Common::Status Load(const std::string& model_uri) {
    LOGS(*session_logger_, INFO) << "Loading model: " << model_uri;
    std::lock_guard<std::mutex> l(session_mutex_);
    if (is_model_loaded_) {  // already loaded
      LOGS(*session_logger_, INFO) << "Model: " << model_uri << " has already been loaded.";
      return Common::Status::OK();
    }

    std::shared_ptr<LotusIR::Model> tmp_model_ptr;
    Common::Status status = LotusIR::Model::Load(model_uri, &tmp_model_ptr);
    if (!status.IsOK()) {
      return status;
    }

    is_model_loaded_ = true;
    model_ = tmp_model_ptr;

    LOTUS_RETURN_IF_ERROR(SaveModelMetadata(*model_.get()));
    LOGS(*session_logger_, INFO) << "Model: " << model_uri << " successfully loaded.";

    return Common::Status::OK();
  }

  Common::Status Initialize() {
    LOGS(*session_logger_, INFO) << "Initializing session.";
    std::lock_guard<std::mutex> l(session_mutex_);
    if (!is_model_loaded_) {
      LOGS(*session_logger_, ERROR) << "Model was not loaded";
      return Common::Status(Common::LOTUS, Common::FAIL, "Model was not loaded.");
    }

    if (is_inited_) {  // already initialized
      LOGS(*session_logger_, INFO) << "Session has already been initialized.";
      return Common::Status::OK();
    }

    LotusIR::Graph* graph = model_->MainGraph();
    LOTUS_RETURN_IF_ERROR(TransformGraph(graph));
    LOTUS_RETURN_IF_ERROR(graph->Resolve());

    // at this point the graph should be in a frozen state
    // hence we set it in the session for use by the executors
    session_state_.SetGraph(graph);

    // All following initialization steps work on the frozen state of
    // graph stored inside session_state.

    LOTUS_RETURN_IF_ERROR(SaveKernelsAndMLValueNameIndexMapping());
    LOTUS_RETURN_IF_ERROR(SaveInitializedTensors());

    // add other per session initialization stuff here before invoking the executor

    // get execution plan
    if (session_options_.enable_sequential_execution) {
      // Why use a unique_ptr here? the only other ways to avoid using a unique_ptr are
      // (1) making a copy or (2) passing a ptr to the private session_state var (p_seq_exec_plan) to CreatePlan.
      // Passing a pointer to a private member variable doesn't seem the right thing to do.
      std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan = std::make_unique<SequentialExecutionPlan>();
      // TODO below line is for testing only. In production use SequentialPlanner::CreatePlan()
      LOTUS_RETURN_IF_ERROR(AllocationPlanner::CreatePlan(session_options_.allocation_planner_type,
                                                          session_state_,
                                                          p_seq_exec_plan.get()));
      session_state_.SetExecutionPlan(std::move(p_seq_exec_plan));
    } else {
      LOTUS_NOT_IMPLEMENTED;
    }

    is_inited_ = true;
    LOGS(*session_logger_, INFO) << "Session successfully initialized.";
    return Status::OK();
  }

  int GetCurrentNumRuns() const {
    return current_num_runs_.load();
  }

  Common::Status Run(const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches) {
    RunOptions run_options;
    return Run(run_options, feeds, output_names, p_fetches);
  }

  static Common::Status ValidateOutputs(const std::vector<std::string>& output_names,
                                        const std::vector<MLValue>* p_fetches) {
    if (!p_fetches) {
      return Common::Status(Common::LOTUS, Common::FAIL, "Output vector pointer is NULL");
    }

    if (!p_fetches->empty() &&
        (output_names.size() != p_fetches->size())) {
      std::ostringstream ostr;
      ostr << "Output vector incorrectly sized: output_names.size(): " << output_names.size()
           << "p_fetches->size(): " << p_fetches->size();
      return Common::Status(Common::LOTUS, Common::FAIL, ostr.str());
    }

    // TODO add more validation here like checking shape of the allocated buffers

    return Common::Status::OK();
  }

  Common::Status Run(const NameMLValMap& feeds,
                     std::vector<MLValue>* p_fetches) {
    RunOptions run_options;
    std::vector<std::string> output_names;
    for (const LotusIR::NodeArg* arg : model_->MainGraph()->GetOutputs()) {
      output_names.push_back(arg->Name());
    }
    return Run(run_options, feeds, output_names, p_fetches);
  }

  Common::Status Run(const RunOptions& run_options,
                     const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches) {
    Common::Status retval;
    try {
      {
        std::lock_guard<std::mutex> l(session_mutex_);
        if (!is_inited_) {
          LOGS(*session_logger_, ERROR) << "Session was not initialized";
          return Common::Status(Common::LOTUS, Common::FAIL, "Session not initialized.");
        }
      }

      // if the output vector is non-empty, ensure that its the same size as the output_names
      LOTUS_RETURN_IF_ERROR(ValidateOutputs(output_names, p_fetches));

      // TODO add instrumentation to measure the time taken for this Run
      if (!run_options.run_tag.empty()) {
        LOGS(*session_logger_, INFO) << "Running with tag: " << run_options.run_tag;
      }

      ++current_num_runs_;

      // TODO should we add this exec to the list of executors? i guess its not needed now?

      // scope of owned_run_logger is just the call to Execute. If Execute ever becomes async we need a different approach
      unique_ptr<Logging::Logger> owned_run_logger;
      auto run_logger = CreateLoggerForRun(run_options, owned_run_logger);

      std::unique_ptr<Executor> p_exec;
      if (session_options_.enable_sequential_execution) {
        p_exec = Executor::NewSequentialExecutor(session_state_, feeds, output_names, *p_fetches, run_logger);
      } else {
        LOTUS_NOT_IMPLEMENTED;
      }

      retval = p_exec->Execute(run_options, feeds, output_names, p_fetches);
    } catch (const std::exception& e) {
      retval = Common::Status(Common::LOTUS, Common::FAIL, e.what());
    }

    --current_num_runs_;
    return retval;
  }

  std::pair<Common::Status, const ModelMetadata*> GetModelMetadata() const {
    {
      std::lock_guard<std::mutex> l(session_mutex_);
      if (!is_model_loaded_) {
        LOGS(*session_logger_, ERROR) << "Model was not loaded";
        return std::make_pair(Common::Status(Common::LOTUS, Common::FAIL, "Model was not loaded."),
                              nullptr);
      }
    }

    return std::make_pair(Common::Status::OK(), &model_metadata_);
  }

  std::pair<Common::Status, const InputDefList*> GetInputs() const {
    {
      std::lock_guard<std::mutex> l(session_mutex_);
      if (!is_model_loaded_) {
        LOGS(*session_logger_, ERROR) << "Model was not loaded";
        return std::make_pair(Common::Status(Common::LOTUS, Common::FAIL, "Model was not loaded."),
                              nullptr);
      }
    }

    return std::make_pair(Common::Status::OK(), &input_def_list_);
  }

  std::pair<Common::Status, const OutputDefList*> GetOutputs() const {
    {
      std::lock_guard<std::mutex> l(session_mutex_);
      if (!is_model_loaded_) {
        LOGS(*session_logger_, ERROR) << "Model was not loaded";
        return std::make_pair(Common::Status(Common::LOTUS, Common::FAIL, "Model was not loaded."),
                              nullptr);
      }
    }

    return std::make_pair(Common::Status::OK(), &output_def_list_);
  }

 private:
  static void GetNodeArgDef(const LotusIR::NodeArg& arg, NodeArgDef& nf) {
    nf.name = arg.Name();
    nf.data_type = *arg.Type();
    nf.shape.clear();
    auto shape = arg.Shape();
    if (nullptr != shape) {
      nf.shape = Utils::GetTensorShapeFromTensorShapeProto(*arg.Shape());
    }
  }

  Common::Status SaveModelMetadata(const LotusIR::Model& model) {
    VLOGS(*session_logger_, 1) << "Saving model metadata";
    const LotusIR::Graph* p_graph = model.MainGraph();
    if (!p_graph) {
      return Common::Status(Common::LOTUS, Common::FAIL, "Got null graph ptr while saving model metadata");
    }

    // save model metadata
    model_metadata_.producer_name = model.ProducerName();
    model_metadata_.description = model.DocString();
    model_metadata_.domain = model.Domain();
    model_metadata_.version = model.ModelVersion();
    model_metadata_.custom_metadata_map = model.MetaData();
    model_metadata_.graph_name = p_graph->Name();

    // save inputs
    auto& inputs = p_graph->GetInputs();
    auto& weights = p_graph->GetAllInitializedTensors();
    input_def_list_.reserve(inputs.size());
    for (const auto& elem : inputs) {
      if (!elem) {
        return Common::Status(Common::LOTUS, Common::FAIL, "Got null input nodearg ptr");
      }
      // skip inputs that are weights
      if (weights.count(elem->Name())) {
        continue;
      }
      NodeArgDef nf;
      GetNodeArgDef(*elem, nf);
      input_def_list_.push_back(nf);
    }

    // save outputs
    auto& outputs = p_graph->GetOutputs();
    output_def_list_.reserve(outputs.size());
    for (const auto& elem : outputs) {
      if (!elem) {
        return Common::Status(Common::LOTUS, Common::FAIL, "Got null output nodearg ptr");
      }
      NodeArgDef nf;
      GetNodeArgDef(*elem, nf);
      output_def_list_.push_back(nf);
    }
    VLOGS(*session_logger_, 1) << "Done saving model metadata";
    return Common::Status::OK();
  }

  // Create a Logger for a single execution if possible. Otherwise use the default logger.
  // If a new logger is created, it will also be stored in new_run_logger,
  // which must remain valid for the duration of the execution.
  // If the default logger is used, new_run_logger will remain empty.
  // The returned value should be used in the execution.
  const Logging::Logger& CreateLoggerForRun(const RunOptions& run_options,
                                            unique_ptr<Logging::Logger>& new_run_logger) {
    const Logging::Logger* run_logger;

    // create a per-run logger if we can
    if (logging_manager_ != nullptr) {
      std::string run_log_id{session_options_.session_logid};

      if (!session_options_.session_logid.empty() && !run_options.run_tag.empty()) {
        run_log_id += ":";
      }

      run_log_id += run_options.run_tag;

      if (run_options.run_log_verbosity_level > 0) {
        new_run_logger = logging_manager_->CreateLogger(run_log_id,
                                                        Logging::Severity::kVERBOSE,
                                                        false,
                                                        run_options.run_log_verbosity_level);
      } else {
        new_run_logger = logging_manager_->CreateLogger(run_log_id);
      }

      run_logger = new_run_logger.get();
      VLOGS(*run_logger, 1) << "Created logger for run with id of " << run_log_id;
    } else {
      // fallback to using default logger. this does NOT have any session or run specific id/tag in it
      run_logger = session_logger_;
      VLOGS(*run_logger, 1) << "Using default logger for run " << run_options.run_tag;
    }

    return *run_logger;
  }

  void InitLogger(Logging::LoggingManager* logging_manager) {
    // create logger for session, using provided logging manager if possible
    if (logging_manager != nullptr) {
      std::string session_logid = !session_options_.session_logid.empty()
                                      ? session_options_.session_logid
                                      : "InferenceSession";  // there's probably a better default...

      if (session_options_.session_log_verbosity_level > 0) {
        owned_session_logger_ = logging_manager->CreateLogger(session_logid,
                                                              Logging::Severity::kVERBOSE,
                                                              false,
                                                              session_options_.session_log_verbosity_level);
      } else {
        owned_session_logger_ = logging_manager->CreateLogger(session_logid);
      }
      session_logger_ = owned_session_logger_.get();
    } else {
      session_logger_ = &Logging::LoggingManager::DefaultLogger();
    }

    session_state_.SetLogger(*session_logger_);
  }

  Common::Status TransformGraph(LotusIR::Graph* graph) {
    bool modified = false;
    for (auto& ep : session_state_.GetExecutionProviders()) {
      // TODO: log which execution provider is transforming the graph and
      // whether modified is true/false.
      Status s = ep->GetTransformer().Apply(graph, &modified);
      if (!s.IsOK()) return s;
    }

    return Common::Status::OK();
  }

  Common::Status SaveInitializedTensors() {
    LOGS(*session_logger_, INFO) << "Saving initialized tensors.";
    const LotusIR::Graph* p_graph = session_state_.GetGraph();
    LOTUS_ENFORCE(p_graph, "Got nullptr for graph from session_state");
    LOTUS_ENFORCE(session_state_.GetNumMLValues() > 0);  // assumes MLValue indexes have been populated
    auto& alloc = AllocatorManager::Instance().GetArena(CPU);
    const LotusIR::InitializedTensorSet& initialized_tensor_set = p_graph->GetAllInitializedTensors();
    for (const auto& entry : initialized_tensor_set) {
      const std::string& name = entry.first;
      int mlvalue_index;
      LOTUS_RETURN_IF_ERROR(session_state_.GetMLValueIdx(name, &mlvalue_index));

      const TensorProto& tensor_proto = entry.second;
      std::unique_ptr<Tensor> p_tensor = nullptr;
      LOTUS_RETURN_IF_ERROR(Lotus::Utils::GetTensorFromTensorProto(tensor_proto, &p_tensor, alloc));
      MLValue mlvalue;
      mlvalue.Init(p_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

      session_state_.AddInitializedTensor(mlvalue_index, mlvalue);
      VLOGS(*session_logger_, 1) << "Added weight with name : " << name << " with index: " << mlvalue_index;
    }

    LOGS(*session_logger_, INFO) << "Done saving initialized tensors";
    return Common::Status::OK();
  }

  // This function does the following:
  // - constructs the kernels and saves them in the session state
  // - builds the MLValue name->idx mapping and saves it in the session state
  // The reason we're doing 2 operations in the same function is so that we iterate
  // through all the nodes only once.
  Common::Status SaveKernelsAndMLValueNameIndexMapping() {
    LOGS(*session_logger_, INFO) << "Saving kernels and MLValue mappings.";
    const LotusIR::Graph* p_graph = session_state_.GetGraph();
    LOTUS_ENFORCE(p_graph, "Got nullptr for graph from session_state");
    int curr_idx = 0;
    for (auto& node : p_graph->Nodes()) {
      // ignore source and sink nodes
      if (p_graph->IsSourceNode(node.Index()) || p_graph->IsSinkNode(node.Index())) {
        continue;
      }

      // construct and save the kernels
      std::unique_ptr<OpKernel> p_op_kernel;
      LOTUS_RETURN_IF_ERROR(CreateOpKernel(node, &p_op_kernel));
      session_state_.AddKernel(node.Index(), std::move(p_op_kernel));

      // build the MLValue->index map
      int unused_var = -1;
      for (gsl::not_null<const LotusIR::NodeArg*> input_def : node.InputDefs()) {
        if (session_state_.GetMLValueIdx(input_def->Name(), &unused_var).IsOK()) {
          continue;
        }
        VLOGS(*session_logger_, 1)
            << "Adding input argument with name: " << input_def->Name() << " to MLValueIndex with index: " << curr_idx;
        session_state_.AddMLValueNameIdx(input_def->Name(), curr_idx++);
      }

      for (gsl::not_null<const LotusIR::NodeArg*> output_def : node.OutputDefs()) {
        if (session_state_.GetMLValueIdx(output_def->Name(), &unused_var).IsOK()) {
          continue;
        }
        VLOGS(*session_logger_, 1)
            << "Adding output argument with name: " << output_def->Name() << " to MLValueIndex with index: " << curr_idx;
        session_state_.AddMLValueNameIdx(output_def->Name(), curr_idx++);
      }
    }

    LOGS(*session_logger_, INFO) << "Done saving kernels and MLValue mappings.";
    return Status::OK();
  }

  Common::Status CreateOpKernel(const LotusIR::Node& node, std::unique_ptr<OpKernel>* p_op_kernel) {
    const std::string& exec_provider_name = node.GetExecutionProvider();
    if (exec_provider_name.empty() || !session_state_.GetExecutionProvider(exec_provider_name)) {
      std::ostringstream error_msg;
      error_msg << "Could not create kernel for node: " << node.Name() << " as there's no execution provider allocated.";
      LOGS(*session_logger_, ERROR) << error_msg.str();
      return Common::Status(Common::LOTUS, Common::FAIL, error_msg.str());
    }

    auto exec_provider = session_state_.GetExecutionProvider(exec_provider_name);
    auto& allocator_info = exec_provider->GetTempSpaceAllocator().Info();
    Common::Status status = KernelRegistry::Instance().CreateKernel(node, allocator_info, exec_provider, p_op_kernel);
    if (!status.IsOK()) {
      LOGS(*session_logger_, ERROR) << "Kernel creation failed for node: "
                                    << node.Name() << " with error: " << status.ErrorMessage();
    }
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

  /// Logging manager if provided.
  Logging::LoggingManager* logging_manager_;

  /// Logger for this session. WARNING: Will contain nullptr if logging_manager_ is nullptr.
  std::unique_ptr<Logging::Logger> owned_session_logger_;

  /// convenience pointer to logger. should always be the same as session_state_.Logger();
  const Logging::Logger* session_logger_;

  // The model served by this inference session instance.
  // Currently this has to be a shared ptr because the Model::Load method
  // returns a shared_ptr only. Ideally factory functions should always return
  // unique_ptr for maximum flexibility. Client can always upgrade it to shared_ptr
  // if they need.
  std::shared_ptr<LotusIR::Model> model_;

  // A set of executors that can run in parallel.
  std::vector<std::unique_ptr<Executor>> executors_;  // TODO do we need this vector?

  // Immutable state for each op in the model. Shared by all executors.
  SessionState session_state_;

  ModelMetadata model_metadata_;
  InputDefList input_def_list_;
  OutputDefList output_def_list_;

  // Environment for this session
  // not used now; we'll need it when we introduce threadpool
  // statically allocated pointer, no need to manage its lifetime.
  //Env* env_;

  // Threadpool for this session
  //thread::ThreadPool thread_pool_; // not used for now; will add it later when implementing RunAsync

  // Number of concurrently running executors
  std::atomic<int> current_num_runs_;

  mutable std::mutex session_mutex_;  // to ensure only one thread can invoke Load/Initialize
  bool is_model_loaded_ = false;      // GUARDED_BY(session_mutex_)
  bool is_inited_ = false;            // GUARDED_BY(session_mutex_)

};  // namespace Lotus

//
// InferenceSession
//
InferenceSession::InferenceSession(const SessionOptions& session_options, Logging::LoggingManager* logging_manager)
    : impl_(std::make_unique<Impl>(session_options, logging_manager)) {
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

Common::Status InferenceSession::Run(const NameMLValMap& feeds,
                                     std::vector<MLValue>* p_fetches) {
  return impl_->Run(feeds, p_fetches);
}
Common::Status InferenceSession::Run(const RunOptions& run_options,
                                     const NameMLValMap& feeds,
                                     const std::vector<std::string>& output_names,
                                     std::vector<MLValue>* p_fetches) {
  return impl_->Run(run_options, feeds, output_names, p_fetches);
}

std::pair<Common::Status, const ModelMetadata*> InferenceSession::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

std::pair<Common::Status, const InputDefList*> InferenceSession::GetInputs() const {
  return impl_->GetInputs();
}

std::pair<Common::Status, const OutputDefList*> InferenceSession::GetOutputs() const {
  return impl_->GetOutputs();
}

int InferenceSession::GetCurrentNumRuns() {
  return impl_->GetCurrentNumRuns();
}

}  // namespace Lotus

#include "core/framework/inference_session.h"

#include <mutex>
#include <sstream>
#include <unordered_set>

#include "core/common/logging/logging.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/customregistry.h"
#include "core/framework/executor.h"
#include "core/framework/execution_frame.h"
#include "core/framework/IOBinding.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/op_kernel_abi_wrapper.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_transformer.h"
#include "core/graph/model.h"
#include "core/graph/tensorutils.h"
#include "core/platform/notification.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/insert_cast_transformer.h"

namespace Lotus {

class InferenceSession::Impl {
 public:
  Impl(const SessionOptions& session_options, Logging::LoggingManager* logging_manager)
      : session_options_{session_options},
        graph_transformation_mgr_{session_options_.max_num_graph_transformation_steps},
        logging_manager_{logging_manager},
        insert_cast_transformer_("CastFloat16Transformer") {
    InitLogger(logging_manager);

    //env_(Env::Default()) {
    //thread_pool_(env_, "Compute", session_options.num_threads) {

    session_state_.SetEnableMemoryPattern(session_options.enable_mem_pattern);
    session_state_.SetProfiler(session_profiler_);
    if (session_options.enable_profiling)
    {
      StartProfiling(session_options.profile_file_prefix);
    }
  }

  Common::Status RegisterExecutionProvider(std::unique_ptr<IExecutionProvider> p_exec_provider) {
    if (p_exec_provider.get() == nullptr) {
      return Status(LOTUS, FAIL, "Received nullptr for exec provider");
    }
    std::string provider_type = p_exec_provider->Type();
    VLOGS(*session_logger_, 1) << "Adding execution provider of type: " << provider_type;
    session_state_.AddExecutionProvider(provider_type, std::move(p_exec_provider));
    return Status::OK();
  }

  Common::Status RegisterGraphTransformer(std::unique_ptr<LotusIR::GraphTransformer> p_graph_transformer) {
    if (p_graph_transformer.get() == nullptr) {
      return Status(LOTUS, FAIL, "Received nullptr for graph transformer");
    }
    return graph_transformation_mgr_.Register(std::move(p_graph_transformer));
  }

  Common::Status RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) {
    if (custom_registry.get() == nullptr) {
      return Status(LOTUS, FAIL, "Received nullptr for custom registry");
    }

    session_state_.GetCustomRegistryManager().RegisterCustomRegistry(custom_registry);
    return Status::OK();
  }

  bool HaslocalSchema() const {
    return session_state_.GetCustomRegistryManager().HasSchema();
  }

  Common::Status Load(const std::string& model_uri) {
    auto tp = session_profiler_.StartTime();
    try {
      LOGS(*session_logger_, INFO) << "Loading model: " << model_uri;
      std::lock_guard<std::mutex> l(session_mutex_);
      if (is_model_loaded_) {  // already loaded
        LOGS(*session_logger_, ERROR) << "This session already contains a loaded model.";
        return Common::Status(Common::LOTUS, Common::MODEL_LOADED, "This session already contains a loaded model.");
      }

      std::shared_ptr<LotusIR::Model> p_tmp_model;
      LOTUS_RETURN_IF_ERROR(LotusIR::Model::Load(model_uri, p_tmp_model, HaslocalSchema() ? &session_state_.GetCustomRegistryManager() : nullptr));
      model_ = p_tmp_model;

      LOTUS_RETURN_IF_ERROR(DoPostLoadProcessing(*model_.get()));

      // all steps complete, mark the model as loaded.
      is_model_loaded_ = true;

      LOGS(*session_logger_, INFO) << "Model: " << model_uri << " successfully loaded.";
    } catch (const std::exception& ex) {
      return Status(LOTUS, FAIL, "Exception during loading: " + std::string(ex.what()));
    } catch (...) {
      LOGS(*session_logger_, ERROR) << "Unknown exception in Load()";
      return Status(LOTUS, RUNTIME_EXCEPTION, "Encountered unknown exception in Load()");
    }
    session_profiler_.EndTimeAndRecordEvent(Profiling::SESSION_EVENT, "model_loading_uri", tp);
    return Common::Status::OK();
  }

  Common::Status Load(const ModelProto& model_proto) {
    auto tp = session_profiler_.StartTime();
    try {
      LOGS(*session_logger_, INFO) << "Loading model using model_proto";
      std::lock_guard<std::mutex> l(session_mutex_);
      if (is_model_loaded_) {  // already loaded
        LOGS(*session_logger_, ERROR) << "This session already contains a loaded model.";
        return Common::Status(Common::LOTUS, Common::MODEL_LOADED, "This session already contains a loaded model.");
      }

      std::shared_ptr<LotusIR::Model> p_tmp_model;
      LOTUS_RETURN_IF_ERROR(LotusIR::Model::Load(model_proto, p_tmp_model, HaslocalSchema() ? &session_state_.GetCustomRegistryManager() : nullptr));
      model_ = p_tmp_model;

      LOTUS_RETURN_IF_ERROR(DoPostLoadProcessing(*model_.get()));

      // all steps complete, mark the model as loaded.
      is_model_loaded_ = true;

      LOGS(*session_logger_, INFO) << "Model successfully loaded.";
    } catch (const std::exception& ex) {
      return Status(LOTUS, FAIL, "Exception during loading: " + std::string(ex.what()));
    } catch (...) {
      LOGS(*session_logger_, ERROR) << "Unknown exception in Load()";
      return Status(LOTUS, RUNTIME_EXCEPTION, "Encountered unknown exception in Load()");
    }
    session_profiler_.EndTimeAndRecordEvent(Profiling::SESSION_EVENT, "model_loading_proto", tp);
    return Status::OK();
  }

  Common::Status Load(std::istream& model_istream) {
    auto tp = session_profiler_.StartTime();
    try {
      LOGS(*session_logger_, INFO) << "Loading model using istream";
      std::lock_guard<std::mutex> l(session_mutex_);
      if (is_model_loaded_) {  // already loaded
        LOGS(*session_logger_, ERROR) << "This session already contains a loaded model.";
        return Common::Status(Common::LOTUS, Common::MODEL_LOADED, "This session already contains a loaded model.");
      }

      ModelProto model_proto;
      const bool result = model_proto.ParseFromIstream(&model_istream);
      if (!result) {
        return Status(LOTUS, INVALID_PROTOBUF, "Failed to load model because protobuf parsing failed.");
      }

      std::shared_ptr<LotusIR::Model> p_tmp_model;
      LOTUS_RETURN_IF_ERROR(LotusIR::Model::Load(model_proto, p_tmp_model, HaslocalSchema() ? &session_state_.GetCustomRegistryManager() : nullptr));
      model_ = p_tmp_model;

      LOTUS_RETURN_IF_ERROR(DoPostLoadProcessing(*model_.get()));

      // all steps complete, mark the model as loaded.
      is_model_loaded_ = true;

      LOGS(*session_logger_, INFO) << "Model successfully loaded.";
    } catch (const std::exception& ex) {
      return Status(LOTUS, FAIL, "Exception during loading: " + std::string(ex.what()));
    } catch (...) {
      LOGS(*session_logger_, ERROR) << "Unknown exception in Load()";
      return Status(LOTUS, RUNTIME_EXCEPTION, "Encountered unknown exception in Load()");
    }
    session_profiler_.EndTimeAndRecordEvent(Profiling::SESSION_EVENT, "model_loading_istream", tp);
    return Common::Status::OK();
  }

  Common::Status Load(std::unique_ptr<LotusIR::Model> p_model) {
    auto tp = session_profiler_.StartTime();
    try {
      LOGS(*session_logger_, INFO) << "Loading model";
      std::lock_guard<std::mutex> l(session_mutex_);
      if (is_model_loaded_) {  // already loaded
        LOGS(*session_logger_, ERROR) << "This session already contains a loaded model.";
        return Common::Status(Common::LOTUS, Common::MODEL_LOADED, "This session already contains a loaded model.");
      }

      model_ = std::move(p_model);

      LOTUS_RETURN_IF_ERROR(DoPostLoadProcessing(*model_.get()));

      // all steps complete, mark the model as loaded.
      is_model_loaded_ = true;

      LOGS(*session_logger_, INFO) << "Model successfully loaded.";
    } catch (const std::exception& ex) {
      return Status(LOTUS, FAIL, "Exception during loading: " + std::string(ex.what()));
    } catch (...) {
      LOGS(*session_logger_, ERROR) << "Unknown exception in Load()";
      return Status(LOTUS, RUNTIME_EXCEPTION, "Encountered unknown exception in Load()");
    }
    session_profiler_.EndTimeAndRecordEvent(Profiling::SESSION_EVENT, "model_loading_p_model", tp);
    return Common::Status::OK();
  }

  Common::Status Initialize() {
    auto tp = session_profiler_.StartTime();
    try {
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

      if (!session_state_.GetExecutionProvider(LotusIR::kCpuExecutionProvider)) {
        // Register default CPUExecutionProvider if user didn't provide it through the Register() calls
        LOGS(*session_logger_, INFO) << "Adding default CPU execution provider.";
        CPUExecutionProviderInfo epi{"CPUExecutionProvider"};
        session_state_.AddExecutionProvider(LotusIR::kCpuExecutionProvider,
                                            std::make_unique<CPUExecutionProvider>(epi));
      }

      LotusIR::Graph* p_graph = model_->MainGraph();
      session_state_.SetGraph(p_graph);

      LOTUS_RETURN_IF_ERROR(TransformGraph(p_graph));
      LOTUS_RETURN_IF_ERROR(p_graph->Resolve());
      LOTUS_RETURN_IF_ERROR(SaveMLValueNameIndexMapping(*p_graph));

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
        LOTUS_NOT_IMPLEMENTED("non sequential execution is not implemented");
      }

      LOTUS_RETURN_IF_ERROR(SaveInitializedTensors(*p_graph));
      p_graph->CleanAllInitializedTensors();  // remove weights from the graph now to save memory

      LOTUS_RETURN_IF_ERROR(SaveKernels(*p_graph));

      is_inited_ = true;

      LOGS(*session_logger_, INFO) << "Session successfully initialized.";
    } catch (const NotImplementedException& ex) {
      LOGS(*session_logger_, ERROR) << "Exception during initialization: " << std::string(ex.what());
      return Status(LOTUS, NOT_IMPLEMENTED, "Exception during initialization: " + std::string(ex.what()));
    } catch (const std::exception& ex) {
      LOGS(*session_logger_, ERROR) << "Exception during initialization: " << std::string(ex.what());
      return Status(LOTUS, FAIL, "Exception during initialization: " + std::string(ex.what()));
    } catch (...) {
      LOGS(*session_logger_, ERROR) << "Unknown exception in Initialize()";
      return Status(LOTUS, RUNTIME_EXCEPTION, "Encountered unknown exception in Initialize()");
    }
    session_profiler_.EndTimeAndRecordEvent(Profiling::SESSION_EVENT, "session_initialization", tp);
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

  Common::Status ValidateInputs(const NameMLValMap& feeds) {
    if (model_input_names_.size() != feeds.size()) {
      return Common::Status(Common::LOTUS, Common::FAIL, "The number of feeds is not same as the number of the model input");
    }

    bool valid = true;
    std::ostringstream invalid_names;
    for (const auto& pair : feeds) {
      if (model_input_names_.find(pair.first) == model_input_names_.end()) {
        valid = false;
        invalid_names << " " << pair.first;
      }
    }

    if (valid) {
      return Common::Status::OK();
    } else {
      std::ostringstream ostr;
      std::for_each(std::begin(model_input_names_), std::end(model_input_names_), [&ostr](const std::string& elem) {
        ostr << elem << " ";
      });
      return Common::Status(Common::LOTUS,
                            Common::INVALID_ARGUMENT,
                            "Invalid Feed Input Names:" + invalid_names.str() +
                                " Valid input names are: " + ostr.str());
    }
  }

  Common::Status ValidateOutputs(const std::vector<std::string>& output_names,
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

    bool valid = true;
    std::ostringstream invalid_names;
    for (const auto& name : output_names) {
      if (model_output_names_.find(name) == model_output_names_.end()) {
        valid = false;
        invalid_names << " " << name;
      }
    }

    if (!valid) {
      std::ostringstream ostr;
      std::for_each(std::begin(model_output_names_), std::end(model_output_names_), [&ostr](const std::string& elem) {
        ostr << elem << " ";
      });
      return Common::Status(Common::LOTUS,
                            Common::INVALID_ARGUMENT,
                            "Invalid Output Names:" + invalid_names.str() +
                                " Valid output names are: " + ostr.str());
    }

    // TODO add more validation here like checking shape of the allocated buffers

    return Common::Status::OK();
  }

  Common::Status Run(const NameMLValMap& feeds,
                     std::vector<MLValue>* p_fetches) {
    RunOptions run_options;
    std::vector<std::string> output_names;
    const LotusIR::Graph* p_graph = model_->MainGraph();
    for (const LotusIR::NodeArg* arg : p_graph->GetOutputs()) {
      output_names.push_back(arg->Name());
    }
    return Run(run_options, feeds, output_names, p_fetches);
  }

  // copies inputs across devices only if required
  Common::Status CopyInputsAcrossDevices(const NameMLValMap& orig_feeds,
                                         NameMLValMap& new_feeds) {
    const LotusIR::Graph* p_graph = session_state_.GetGraph();
    LOTUS_ENFORCE(p_graph);
    int count_feeds = 0;
    auto& weights_map = p_graph->GetAllInitializedTensors();
    for (auto& node : p_graph->Nodes()) {  // TODO optimize this
      if (count_feeds == orig_feeds.size()) {
        break;
      }
      for (auto* arg : node.InputDefs()) {
        if (!arg->Exists() ||
            arg->Name().empty() ||
            !orig_feeds.count(arg->Name()) ||
            weights_map.count(arg->Name())) {
          continue;
        }

        ++count_feeds;
        auto& input_name = arg->Name();
        auto& node_provider_type = node.GetExecutionProviderType();
        auto& mlvalue = orig_feeds.at(input_name);
        if (!mlvalue.IsTensor()) {
          // copying not supported for non-tensor types
          new_feeds[input_name] = mlvalue;
          continue;
        }
        auto& input_tensor = mlvalue.Get<Tensor>();
        auto& input_tensor_loc = input_tensor.Location();
        auto* p_input_provider = session_state_.GetExecutionProvider(input_tensor_loc);
        if (!p_input_provider) {
          p_input_provider = session_state_.GetExecutionProvider(LotusIR::kCpuExecutionProvider);
        }
        LOTUS_ENFORCE(p_input_provider);

        auto input_provider_type = p_input_provider->Type();
        if (input_provider_type == node_provider_type) {
          new_feeds[input_name] = mlvalue;
          continue;
        }

        auto* node_provider = session_state_.GetExecutionProvider(node_provider_type);
        LOTUS_ENFORCE(node_provider);
        MLValue new_mlvalue;
        LOTUS_RETURN_IF_ERROR(AllocateHelper(node_provider_type, mlvalue, new_mlvalue));
        auto* new_tensor = new_mlvalue.GetMutable<Tensor>();
        auto* node_exec_provider = session_state_.GetExecutionProvider(node_provider_type);
        LOTUS_ENFORCE(node_exec_provider);

        // our CPU exec provider doesn't support copy from GPU->CPU
        if (node_provider_type != LotusIR::kCpuExecutionProvider) {
          LOTUS_RETURN_IF_ERROR(node_exec_provider->CopyTensor(input_tensor, *new_tensor));
        } else {
          LOTUS_RETURN_IF_ERROR(p_input_provider->CopyTensor(input_tensor, *new_tensor));
        }

        new_feeds[input_name] = new_mlvalue;
      };
    }
    LOTUS_ENFORCE(orig_feeds.size() == new_feeds.size());
    return Status::OK();
  }

  static std::pair<bool, size_t> Contains(const std::vector<std::string>& output_names, const std::string& name) {
    auto it = std::find(std::begin(output_names), std::end(output_names), name);
    if (it == output_names.end()) {
      return {false, 0};
    }
    return {true, it - output_names.begin()};
  }

  // ensures pre-allocated outputs match the node providers
  Common::Status MatchOutputsWithProviders(const std::vector<std::string>& output_names,
                                           std::vector<MLValue>& fetches,
                                           std::vector<MLValue>& new_fetches) {
    if (fetches.empty()) {
      fetches.resize(output_names.size());
    }
    new_fetches.resize(output_names.size());

    std::set<std::string> seen_outputs;
    const LotusIR::Graph* p_graph = session_state_.GetGraph();
    LOTUS_ENFORCE(p_graph);
    std::pair<bool, size_t> found;
    for (auto& node : p_graph->Nodes()) {  // TODO optimize this
      if (seen_outputs.size() == fetches.size()) {
        break;
      }
      for (auto* arg : node.OutputDefs()) {
        if (!arg->Exists() ||
            arg->Name().empty() ||
            !(found = Contains(output_names, arg->Name())).first) {
          continue;
        }

        seen_outputs.insert(arg->Name());
        size_t idx = found.second;
        MLValue orig_mlvalue = fetches[idx];
        if (orig_mlvalue.IsAllocated()) {
          if (!orig_mlvalue.IsTensor()) {
            new_fetches[idx] = fetches[idx];
            continue;
          }

          auto& node_provider_type = node.GetExecutionProviderType();
          auto& orig_tensor = orig_mlvalue.Get<Tensor>();
          auto& orig_tensor_loc = orig_tensor.Location();
          auto* tensor_provider = session_state_.GetExecutionProvider(orig_tensor_loc);
          if (!tensor_provider) {
            tensor_provider = session_state_.GetExecutionProvider(LotusIR::kCpuExecutionProvider);
          }
          auto tensor_provider_type = tensor_provider->Type();
          if (node_provider_type == tensor_provider_type) {
            new_fetches[idx] = fetches[idx];
            continue;
          } else {
            // leave the new_fetches[idx] as it is since it'll get allocated on the appropriate provider by the
            // op kernel context when requested
            continue;
          }
        } else {
          new_fetches[idx] = fetches[idx];
          continue;
        }
      }
    }

    // if we've already seen all the outputs requested just return
    if (seen_outputs.size() == output_names.size()) {
      return Status::OK();
    }

    // handle the case when a constant is an output but has been folded into a weight
    // and hence it doesn't show up in any of the OutputDefs before
    // assume that the weight has already been placed in the appropriate device before
    auto& defs = p_graph->GetOutputs();
    for (auto& one_def : defs) {
      if (!one_def->Exists() ||
          one_def->Name().empty() ||
          seen_outputs.count(one_def->Name()) ||
          !(found = Contains(output_names, one_def->Name())).first) {
        continue;
      }

      auto& def_name = one_def->Name();
      size_t idx = found.second;
      int mlvalue_idx;
      LOTUS_RETURN_IF_ERROR(session_state_.GetMLValueIdx(def_name, &mlvalue_idx));
      auto& weights = session_state_.GetInitializedTensors();
      if (!weights.count(mlvalue_idx)) {
        LOGS(*session_logger_, INFO) << "Output with name " << def_name << " is not a weight.";
        continue;
      }
      seen_outputs.insert(def_name);
      auto weight = session_state_.GetInitializedTensors().at(mlvalue_idx);
      new_fetches[idx] = weight;
    }

    LOTUS_ENFORCE(seen_outputs.size() == output_names.size());  // make sure we've seen all outputs
    return Status::OK();
  }

  Common::Status AllocateHelper(LotusIR::ProviderType provider_type,
                                const MLValue& fetched_mlvalue,
                                MLValue& output_mlvalue) {
    auto* p_provider = session_state_.GetExecutionProvider(provider_type);
    LOTUS_ENFORCE(p_provider);
    auto allocator = p_provider->GetAllocator();
    LOTUS_ENFORCE(allocator != nullptr);
    auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();
    void* buffer = allocator->Alloc(fetched_tensor.DataType()->Size() * fetched_tensor.Shape().Size());
    LOTUS_ENFORCE(buffer);
    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(fetched_tensor.DataType(),
                                                                fetched_tensor.Shape(),
                                                                buffer,
                                                                allocator->Info(),
                                                                allocator);
    output_mlvalue.Init(p_tensor.release(),
                        DataTypeImpl::GetType<Tensor>(),
                        DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

    return Status::OK();
  }

  // copies outputs across devices only if required
  Common::Status CopyOutputsAcrossDevices(std::vector<MLValue>& fetches,
                                          std::vector<MLValue>& user_fetches) {
    for (size_t idx = 0, end = fetches.size(); idx < end; ++idx) {
      auto& fetched_mlvalue = fetches[idx];
      if (!fetched_mlvalue.IsTensor()) {
        user_fetches[idx] = fetched_mlvalue;
        continue;
      }

      auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();
      auto& fetched_tensor_location = fetched_tensor.Location();
      auto* p_fetched_provider = session_state_.GetExecutionProvider(fetched_tensor_location);
      if (!p_fetched_provider) {
        p_fetched_provider = session_state_.GetExecutionProvider(LotusIR::kCpuExecutionProvider);
      }
      LOTUS_ENFORCE(p_fetched_provider);
      auto fetched_provider_type = p_fetched_provider->Type();

      auto& output_mlvalue = user_fetches[idx];
      if (!output_mlvalue.IsAllocated()) {
        if (fetched_provider_type != LotusIR::kCpuExecutionProvider) {
          LOTUS_RETURN_IF_ERROR(AllocateHelper(LotusIR::kCpuExecutionProvider,
                                               fetched_mlvalue,
                                               output_mlvalue));
        } else {
          user_fetches[idx] = fetched_mlvalue;
          continue;
        }
      }

      Tensor* p_output_tensor = output_mlvalue.GetMutable<Tensor>();
      auto& output_tensor_loc = p_output_tensor->Location();
      auto* p_output_provider = session_state_.GetExecutionProvider(output_tensor_loc);
      if (!p_output_provider) {
        p_output_provider = session_state_.GetExecutionProvider(LotusIR::kCpuExecutionProvider);
      }
      LOTUS_ENFORCE(p_output_provider);
      auto output_provider_type = p_output_provider->Type();

      if (output_provider_type == fetched_provider_type) {
        user_fetches[idx] = fetched_mlvalue;
        continue;
      }

      // our CPU exec provider doesn't support copy from GPU->CPU
      if (fetched_provider_type != LotusIR::kCpuExecutionProvider) {
        LOTUS_RETURN_IF_ERROR(p_fetched_provider->CopyTensor(fetched_tensor, *p_output_tensor));
      } else {
        LOTUS_RETURN_IF_ERROR(p_output_provider->CopyTensor(fetched_tensor, *p_output_tensor));
      }
    }
    return Status::OK();
  }

  Common::Status Run(const RunOptions& run_options0,
                     const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches) {
    auto tp = session_profiler_.StartTime();
    Common::Status retval;
    const RunOptions run_options(run_options0);
    try {
      {
        std::lock_guard<std::mutex> l(session_mutex_);
        if (!is_inited_) {
          LOGS(*session_logger_, ERROR) << "Session was not initialized";
          return Common::Status(Common::LOTUS, Common::FAIL, "Session not initialized.");
        }
      }

      LOTUS_RETURN_IF_ERROR(ValidateInputs(feeds));

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

      NameMLValMap copied_feeds;
      LOTUS_RETURN_IF_ERROR(CopyInputsAcrossDevices(feeds, copied_feeds));

      std::vector<MLValue> new_fetches;
      LOTUS_RETURN_IF_ERROR(MatchOutputsWithProviders(output_names, *p_fetches, new_fetches));

      std::unique_ptr<Executor> p_exec;
      if (session_options_.enable_sequential_execution) {
        p_exec = Executor::NewSequentialExecutor(session_state_, copied_feeds, output_names, new_fetches, run_logger);
      } else {
        LOTUS_NOT_IMPLEMENTED("non sequential execution is not implemented");
      }

      retval = p_exec->Execute(run_options, copied_feeds, output_names, &new_fetches);
      if (retval.IsOK()) {
        retval = CopyOutputsAcrossDevices(new_fetches, *p_fetches);
      }
    } catch (const std::exception& e) {
      retval = Common::Status(Common::LOTUS, Common::FAIL, e.what());
    } catch (...) {
      retval = Status(LOTUS, RUNTIME_EXCEPTION, "Encountered unknown exception in Run()");
    }

    --current_num_runs_;
    session_profiler_.EndTimeAndRecordEvent(Profiling::SESSION_EVENT, "model_run", tp);
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

  Common::Status NewIOBinding(LotusIR::ProviderType provider_type, std::unique_ptr<IOBinding>* io_binding) {
    IExecutionProvider* p_exec_provider = session_state_.GetExecutionProvider(provider_type);
    if (!p_exec_provider) {
      return Status(LOTUS, FAIL, "You did not register this execution provider before.");
    }
    *io_binding = std::unique_ptr<IOBinding>(new IOBinding(p_exec_provider, session_logger_));  // private constructor, can't use make_unique
    return Status::OK();
  }

  Common::Status Run(const RunOptions& run_options, IOBinding& io_binding) {
    // TODO should Run() call io_binding.SynchronizeInputs() or should it let the callers do it?
    // io_binding.SynchronizeInputs();
    return Run(run_options, io_binding.feeds_, io_binding.output_names_, &io_binding.outputs_);
  }

  Common::Status Run(IOBinding& io_binding) {
    RunOptions run_options;
    return Run(run_options, io_binding);
  }

  void StartProfiling(const std::string& file_prefix) {
    std::stringstream ss;
    ss << file_prefix << "_" <<GetCurrentTimeString() << ".json"; 
    session_profiler_.StartProfiling(session_logger_, ss.str());
  }

  std::string EndProfiling() {
    if (is_model_loaded_) {
      return session_profiler_.WriteProfileData();
    } else {
      LOGS(*session_logger_, ERROR) << "Could not write a profile because no model was loaded.";
      return std::string();
    }
  }

 private:
  // assumes model has already been loaded before
  Common::Status DoPostLoadProcessing(LotusIR::Model& model) {
    // TODO add other post load processing here
    Common::Status status = SaveModelMetadata(model);
    return status;
  }

  Common::Status SaveModelMetadata(const LotusIR::Model& model) {
    VLOGS(*session_logger_, 1) << "Saving model metadata";
    const LotusIR::Graph* p_graph = model.MainGraph();

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
      input_def_list_.push_back(elem);
      model_input_names_.insert(elem->Name());
    }

    // save outputs
    auto& outputs = p_graph->GetOutputs();
    output_def_list_.reserve(outputs.size());
    for (const auto& elem : outputs) {
      if (!elem) {
        return Common::Status(Common::LOTUS, Common::FAIL, "Got null output nodearg ptr");
      }
      output_def_list_.push_back(elem);
      model_output_names_.insert(elem->Name());
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
    // first apply the default/system/basic transformations
    LOTUS_RETURN_IF_ERROR(graph_transformation_mgr_.ApplyAll(graph));

    // now apply the transformations from the execution providers
    bool modified = false;
    for (auto& ep : session_state_.GetExecutionProviders()) {
      // TODO: log which execution provider is transforming the graph and
      // whether modified is true/false.
      LOTUS_RETURN_IF_ERROR(ep->GetTransformer().Apply(graph, &modified));
    }

    for (auto registry : session_state_.GetCustomRegistryManager().GetAllKernelRegistries()) {
      insert_cast_transformer_.AddKernelRegistry(registry);
    }

    insert_cast_transformer_.AddKernelRegistry(&KernelRegistry::Instance());

    LOTUS_RETURN_IF_ERROR(insert_cast_transformer_.Apply(graph, &modified));

    return Common::Status::OK();
  }

  Common::Status DeserializeTensorProto(const AllocatorInfo& alloc_info, const TensorProto& tensor_proto, MLValue& mlvalue, void* preallocated, size_t preallocated_size) {
    std::unique_ptr<Tensor> p_tensor;
    auto alloc_ptr = session_state_.GetAllocator(alloc_info);
    if (!alloc_ptr) {
      return Status(LOTUS, FAIL, "Failed to get allocator for alloc_info: " + alloc_info.ToString());
    }

    if (alloc_info.name == CPU || alloc_info.mem_type == kMemTypeCPU) {
      // deserilize directly to CPU tensor
      LOTUS_RETURN_IF_ERROR(Lotus::Utils::GetTensorFromTensorProto(tensor_proto, &p_tensor, alloc_ptr, preallocated, preallocated_size));
    } else {
      // deserialize to CPU first for non-CPU allocator, then alloc and copy
      AllocatorPtr deserialize_alloc_ptr;
      std::unique_ptr<Tensor> p_deserialize_tensor;
      deserialize_alloc_ptr = session_state_.GetExecutionProvider(LotusIR::kCpuExecutionProvider)->GetAllocator();
      LOTUS_RETURN_IF_ERROR(Lotus::Utils::GetTensorFromTensorProto(tensor_proto, &p_deserialize_tensor, deserialize_alloc_ptr));

      if (preallocated && preallocated_size != p_deserialize_tensor->Size())
        return Status(LOTUS, FAIL, "The buffer planner is not consistent with tensor buffer size");

      IExecutionProvider* provider = session_state_.GetExecutionProvider(alloc_info);
      LOTUS_ENFORCE(provider != nullptr);
      p_tensor.reset(new Tensor(p_deserialize_tensor->DataType(),
                                p_deserialize_tensor->Shape(),
                                preallocated ? preallocated : static_cast<void*>(alloc_ptr->Alloc(p_deserialize_tensor->Size())),
                                alloc_info,
                                preallocated ? nullptr : alloc_ptr));  // no deleter for preallocated
      Status copy_status = provider->CopyTensor(*p_deserialize_tensor, *p_tensor);
      if (!copy_status.IsOK()) {
        if (copy_status.ErrorMessage().empty()) {
          // The windows execution provider does not return any error message today for CopyTensor since it is
          // not implemented yet. That's the reason we're adding our own error message so that we can debug better.
          return Status(copy_status.Category(),
                        copy_status.Code(),
                        "Failed to copy tensor to execution provider: " + provider->Type());
        } else {
          return copy_status;
        }
      }
    }

    mlvalue.Init(p_tensor.release(),
                 DataTypeImpl::GetType<Tensor>(),
                 DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

    return Common::Status::OK();
  }

  Common::Status SaveInitializedTensorsWithSeperateBuffer(const LotusIR::Graph& graph) {
    LOGS(*session_logger_, INFO) << "Saving initialized tensors.";
    LOTUS_ENFORCE(session_state_.GetNumMLValues() > 0);  // assumes MLValue indexes have been populated

    auto* p_execution_plan = session_state_.GetExecutionPlan();
    LOTUS_ENFORCE(p_execution_plan);  // execution plan must be ready.

    const LotusIR::InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    for (const auto& entry : initialized_tensor_set) {
      const std::string& name = entry.first;
      int mlvalue_index;
      LOTUS_RETURN_IF_ERROR(session_state_.GetMLValueIdx(name, &mlvalue_index));
      VLOGS(*session_logger_, 1) << "About to add weight with name: " << name << " and index: " << mlvalue_index;
      auto& location = p_execution_plan->allocation_plan[mlvalue_index].location;
      MLValue mlvalue;
      LOTUS_RETURN_IF_ERROR(DeserializeTensorProto(location, *(entry.second), mlvalue, nullptr, 0));
      session_state_.AddInitializedTensor(mlvalue_index, mlvalue);
      VLOGS(*session_logger_, 1) << "Added weight with name : " << name << " with index: " << mlvalue_index;
    }

    LOGS(*session_logger_, INFO) << "Done saving initialized tensors";
    return Common::Status::OK();
  }

  Common::Status SaveInitializedTensorsWithMemPattern(const LotusIR::Graph& graph) {
    LOGS(*session_logger_, INFO) << "Saving initialized tensors.";
    LOTUS_ENFORCE(session_state_.GetNumMLValues() > 0);  // assumes MLValue indexes have been populated

    auto execution_plan = session_state_.GetExecutionPlan();
    LOTUS_ENFORCE(execution_plan);  // execution plan must be ready.

    MLValuePatternPlanner planner(session_state_);
    //1. first plan the memory
    const LotusIR::InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    for (const auto& entry : initialized_tensor_set) {
      const std::string& name = entry.first;
      int mlvalue_index;
      LOTUS_RETURN_IF_ERROR(session_state_.GetMLValueIdx(name, &mlvalue_index));

      const TensorProto& tensor_proto = *(entry.second);
      LOTUS_RETURN_IF_ERROR(Lotus::Utils::TraceTensorAllocFromTensorProto(mlvalue_index, tensor_proto, &planner));
    }
    //2. allocate weight buffer on different locations
    MemoryPatternGroup mem_patterns;
    LOTUS_RETURN_IF_ERROR(planner.GeneratePatterns(&mem_patterns));
    for (int i = 0; i < mem_patterns.locations.size(); i++) {
      auto& location = mem_patterns.locations[i];
      LOTUS_ENFORCE(weights_buffers_.find(location) == weights_buffers_.end());
      auto alloc = session_state_.GetAllocator(location);
      if (!alloc)
        return Status(LOTUS, FAIL, "Failed to get allocator for location: " + location.ToString());
      void* buffer = mem_patterns.patterns[i].peak_size() > 0 ? alloc->Alloc(mem_patterns.patterns[i].peak_size()) : nullptr;
      weights_buffers_[location] = BufferUniquePtr(buffer, alloc);
    }
    //3. create weight tensors based on weights buffer
    for (const auto& entry : initialized_tensor_set) {
      const std::string& name = entry.first;
      int mlvalue_index;
      LOTUS_RETURN_IF_ERROR(session_state_.GetMLValueIdx(name, &mlvalue_index));
      const TensorProto& tensor_proto = *(entry.second);

      auto& location = execution_plan->allocation_plan[mlvalue_index].location;
      auto it = weights_buffers_.find(location);
      if (it == weights_buffers_.end())
        return Status(LOTUS, FAIL, "Weight buffer not found");

      auto pattern = mem_patterns.GetPatterns(location);
      auto block = pattern->GetBlock(mlvalue_index);
      MLValue mlvalue;
      // if block is not found, means this mlvalue is not traced
      // fall back to allocate seperate buffer.
      if (!block) {
        LOTUS_RETURN_IF_ERROR(DeserializeTensorProto(location, tensor_proto, mlvalue, nullptr, 0));
      } else {
        LOTUS_RETURN_IF_ERROR(DeserializeTensorProto(location, tensor_proto, mlvalue, (uint8_t*)it->second.get() + block->offset_, block->size_));
      }

      session_state_.AddInitializedTensor(mlvalue_index, mlvalue);
      VLOGS(*session_logger_, 1) << "Added weight with name : " << name << " with index: " << mlvalue_index;
    }

    LOGS(*session_logger_, INFO) << "Done saving initialized tensors";
    return Common::Status::OK();
  }

  Common::Status SaveInitializedTensors(const LotusIR::Graph& graph) {
    auto execution_plan = session_state_.GetExecutionPlan();
    // if we enable the meory pattern and already have the execution plan
    // go with mem pattern approach, which will allocate a big chunk for all
    // the weights.
    if (session_state_.GetEnableMemoryPattern() && execution_plan) {
      return SaveInitializedTensorsWithMemPattern(graph);
    } else {
      return SaveInitializedTensorsWithSeperateBuffer(graph);
    }
  }

  // This function does the following:
  // - builds the MLValue name->idx mapping and saves it in the session state
  Common::Status SaveMLValueNameIndexMapping(const LotusIR::Graph& graph) {
    LOGS(*session_logger_, INFO) << "Saving MLValue mappings.";
    int curr_idx = 0;

    for (auto& node : graph.Nodes()) {
      // ignore source and sink nodes
      if (graph.IsSourceNode(node.Index()) || graph.IsSinkNode(node.Index())) {
        continue;
      }

      // build the MLValue->index map
      for (gsl::not_null<const LotusIR::NodeArg*> input_def : node.InputDefs()) {
        VLOGS(*session_logger_, 1)
            << "Adding input argument with name: " << input_def->Name() << " to MLValueIndex with index: " << curr_idx;
        if (input_def->Exists()) {
          session_state_.AddMLValueNameIdx(input_def->Name(), curr_idx++);
        }
      }

      for (gsl::not_null<const LotusIR::NodeArg*> output_def : node.OutputDefs()) {
        VLOGS(*session_logger_, 1)
            << "Adding output argument with name: " << output_def->Name() << " to MLValueIndex with index: " << curr_idx;
        if (output_def->Exists()) {
          session_state_.AddMLValueNameIdx(output_def->Name(), curr_idx++);
        }
      }
    }

    // allocate MLValue for graph outputs when coming from initializers
    for (const auto& output : graph.GetOutputs()) {
      if (output->Exists()) {
        session_state_.AddMLValueNameIdx(output->Name(), curr_idx++);
      }
    }

    LOGS(*session_logger_, INFO) << "Done saving MLValue mappings.";
    return Status::OK();
  }

  // This function does the following:
  // - constructs the kernels and saves them in the session state
  Common::Status SaveKernels(const LotusIR::Graph& graph) {
    LOGS(*session_logger_, INFO) << "Saving kernels.";
    session_state_.SetKernelVectorSize(graph.MaxNodeIndex());
    for (auto& node : graph.Nodes()) {
      // ignore source and sink nodes
      if (graph.IsSourceNode(node.Index()) || graph.IsSinkNode(node.Index())) {
        continue;
      }
      // construct and save the kernels
      std::unique_ptr<OpKernel> p_op_kernel;
      LOTUS_RETURN_IF_ERROR(CreateOpKernel(node, &p_op_kernel));
      session_state_.AddKernel(node.Index(), std::move(p_op_kernel));
    }

    LOGS(*session_logger_, INFO) << "Done saving kernels.";
    return Status::OK();
  }

  Common::Status CreateOpKernel(const LotusIR::Node& node, std::unique_ptr<OpKernel>* p_op_kernel) {
    LotusIR::ProviderType exec_provider_name = node.GetExecutionProviderType();
    if (exec_provider_name.empty() || !session_state_.GetExecutionProvider(exec_provider_name)) {
      std::ostringstream error_msg;
      error_msg << "Could not create kernel for node: " << node.Name() << " as there's no execution provider allocated.";
      LOGS(*session_logger_, ERROR) << error_msg.str();
      return Common::Status(Common::LOTUS, Common::FAIL, error_msg.str());
    }

    auto exec_provider = session_state_.GetExecutionProvider(exec_provider_name);
    Common::Status status = CreateOpKernelInternal(node, exec_provider, p_op_kernel);
    if (!status.IsOK()) {
      LOGS(*session_logger_, ERROR) << "Kernel creation failed for node: "
                                    << node.Name() << " with error: " << status.ErrorMessage();
    }
    return status;
  }

  Common::Status CreateOpKernelInternal(const LotusIR::Node& node, IExecutionProvider* exec_provider, std::unique_ptr<OpKernel>* p_op_kernel) {
    Common::Status status = session_state_.GetCustomRegistryManager().CreateKernel(node, exec_provider, session_state_, p_op_kernel);

    if (status.IsOK()) {
      return status;
    }

    return KernelRegistry::Instance().CreateKernel(node, exec_provider, session_state_, p_op_kernel);
  }

  Common::Status WaitForNotification(Notification* p_executor_done, int64 timeout_in_ms) {
    if (timeout_in_ms > 0) {
      LOTUS_NOT_IMPLEMENTED(__FUNCTION__, "timeout_in_ms >0 is not supported");  // TODO
    } else {
      p_executor_done->WaitForNotification();
    }

    return Status::OK();
  }

  const SessionOptions session_options_;

  LotusIR::GraphTransformerManager graph_transformation_mgr_;

  /// Logging manager if provided.
  Logging::LoggingManager* logging_manager_;

  /// Logger for this session. WARNING: Will contain nullptr if logging_manager_ is nullptr.
  std::unique_ptr<Logging::Logger> owned_session_logger_;

  /// convenience pointer to logger. should always be the same as session_state_.Logger();
  const Logging::Logger* session_logger_;

  // Profiler for this session.
  Profiling::Profiler session_profiler_;

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

  // names of model inputs and outputs used for quick validation.
  std::unordered_set<std::string> model_input_names_;
  std::unordered_set<std::string> model_output_names_;

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

  std::map<AllocatorInfo, BufferUniquePtr> weights_buffers_;
  InsertCastTransformer insert_cast_transformer_;
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

Common::Status InferenceSession::Load(std::unique_ptr<LotusIR::Model> p_model) {
  return impl_->Load(std::move(p_model));
}

Common::Status InferenceSession::Load(std::istream& model_istream) {
  return impl_->Load(model_istream);
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

void InferenceSession::StartProfiling(const std::string& file_prefix) {
  impl_->StartProfiling(file_prefix);
}

std::string InferenceSession::EndProfiling() {
  return impl_->EndProfiling();
}

Common::Status InferenceSession::RegisterExecutionProvider(std::unique_ptr<IExecutionProvider> p_exec_provider) {
  return impl_->RegisterExecutionProvider(std::move(p_exec_provider));
}

Common::Status InferenceSession::RegisterGraphTransformer(std::unique_ptr<LotusIR::GraphTransformer> p_graph_transformer) {
  return impl_->RegisterGraphTransformer(std::move(p_graph_transformer));
}

Common::Status InferenceSession::RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) {
  return impl_->RegisterCustomRegistry(custom_registry);
}

Common::Status InferenceSession::Load(const ModelProto& model_proto) {
  return impl_->Load(model_proto);
}

Common::Status InferenceSession::NewIOBinding(LotusIR::ProviderType provider_type, std::unique_ptr<IOBinding>* io_binding) {
  return impl_->NewIOBinding(provider_type, io_binding);
}

Common::Status InferenceSession::Run(const RunOptions& run_options, IOBinding& io_binding) {
  return impl_->Run(run_options, io_binding);
}

Common::Status InferenceSession::Run(IOBinding& io_binding) {
  return impl_->Run(io_binding);
}
}  // namespace Lotus

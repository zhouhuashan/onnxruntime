// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/session.h"
#include "core/framework/data_types.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/common/profiler.h"
#include "core/framework/execution_providers.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/session_state.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/framework/path_lib.h"

#ifdef USE_EIGEN_THREADPOOL
#include <unsupported/Eigen/CXX11/ThreadPool>
#endif

namespace ONNX_NAMESPACE {
class ModelProto;
}

namespace onnxruntime {
class IExecutionProvider;
class Graph;
class GraphTransformer;
class Model;
class IOBinding;
class Notification;
class IExecutor;
class TaskThreadPool;

class InferenceSession : public Session {
 public:
  /**
    * Register an execution provider. If you've one to register, call this before invoking Initialize().
    * The order of invocation indicates the preference order as well. In other words call this method 
    * on your most preferred execution provider first followed by the less preferred ones.
    * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
    * @return OK if success.
    */
  common::Status RegisterExecutionProvider(std::unique_ptr<IExecutionProvider> p_exec_provider) override;

  /**
    * Register a graph transformer. If you've one to register, call this before invoking Initialize().
    * Calling this API is optional.
    * @param[in] - providers Optional. If providers is non-empty this transformer will only to 
      applied to nodes which are assigned to given providers.
    * @param[in] - level Optional. Level to which this transformer should be registered. Default is set to 2.
    * @return OK if success.
    */
  common::Status RegisterGraphTransformer(std::unique_ptr<onnxruntime::GraphTransformer> p_graph_transformer,
                                          const std::vector<std::string>& providers,
                                          const uint32_t& level) override;

  /**
    * Enable a custom set of transformers. Call this before invoking Initialize().
    * Calling this API is optional.
    * When this list is provided ORT ignores the levels set in session options.
    * @return OK if success.
    */
  common::Status AddCustomTransformerList(const std::vector<std::string>& transformers_to_enable) override;

  common::Status AddCustomOpDomains(const std::vector<OrtCustomOpDomain*>& ops) override;

  /**
    * Register a custom registry for operator schema and kernels.  If you've one to register, 
    * call this before invoking Initialize().
    * The order of invocation indicates the reversed preference order: Register your most 
    * preferred registry at the end.
    * Calling this API is optional.
    * @return OK if success.
    */
  common::Status RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) override;

  /**
    * Load an ONNX model.
    * @param model_uri absolute path of the model file.
    * @return OK if success.
    */
  common::Status Load(const std::string& model_uri) override;
#ifdef _WIN32
  common::Status Load(const std::wstring& model_uri) override;
#endif
  /**
    * Load an ONNX model.
    * @param istream object of the model.
    * @return OK if success.
    */
  common::Status Load(std::istream& model_istream) override;

  /**
    * Initializes a previously loaded model. Initialization includes but is not
    * limited to graph transformations, construction of kernels, etc.
    * This method assumes that a method has been loaded previously.
    * @return OK if success
    */
  common::Status Initialize() override;

  common::Status Run(const RunOptions& run_options,
                     const std::vector<std::string>& feed_names,
                     const std::vector<MLValue>& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches) override;

  /**
    * Run a pre-loaded and pre-intialized model.
    * Multiple threads are allowed to run this function; hence its thread-safe.
    * @param feeds named inputs owned by client code and should not be changed during
    *        execution of this function.
    * @param output_names output names
    * @param p_fetches output values in the order specified by output_names.
    *        This should not be changed during execution of this function.
    * @return OK if success.
    */
  common::Status Run(const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches) override {
    return Run({}, feeds, output_names, p_fetches);
  }

  /**
    * See Run(const NameMLValMap& feeds, const std::vector<std::string>& output_names, std::vector<MLValue>* p_fetches)
    * for details.
    * @param run_options use this to tune the Run call to your needs.
    */
  common::Status Run(const RunOptions& run_options,
                     const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches) override;

  /**
  * Creates a new binding object for binding inputs and outputs.
  * @param provider_type specifies the location where the inputs need to be potentially copied. 
  * See IOBinding class for more info.
  */
  common::Status NewIOBinding(std::unique_ptr<IOBinding>* io_binding) override;

  common::Status Run(const RunOptions& run_options, IOBinding& io_binding) override;
  common::Status Run(IOBinding& io_binding) override;

  /**
    * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
    * @note lifetime of the returned pointer is valid as long as the Session object is live.
    */
  std::pair<common::Status, const ModelMetadata*> GetModelMetadata() const override;

  /**
    * Get all input definitions of the model. This does not include weights. Use this
    * to get the name/type/shapes of the inputs.
    * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
    * @note lifetime of the returned pointer is valid as long as the Session object is live.
    */
  std::pair<common::Status, const InputDefList*> GetModelInputs() const override;

  /**
    * Get all output definitions of the model. Use this to get the name/type/shapes of the outputs.
    * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
    * @note lifetime of the returned pointer is valid as long as the Session object is live.
    */
  std::pair<common::Status, const OutputDefList*> GetModelOutputs() const override;

  /**
    * Get the current number of in-progress concurrent Run calls.
    */
  int GetCurrentNumRuns() const override {
    return current_num_runs_.load();
  }

  /**
    * Start profiling on this inference session. This simply turns on profiling events to be 
    * recorded. A corresponding EndProfiling has to follow to write profiling data to a file.
    *@param file_prefix is the prefix of the profile file. It can include a directory path. 
    */
  void StartProfiling(const std::string& file_prefix) override;
#ifdef _WIN32
  void StartProfiling(const std::wstring& file_prefix) override;
#endif
  /**
    * Start profiling on this inference session. This simply turns on profiling events to be
    * recorded. A corresponding EndProfiling has to follow to send profiling events through the logger's ISink.
    *@param logger_ptr is pointer to the logger where profiling events will be sent to.
    */
  void StartProfiling(const logging::Logger* logger_ptr) override;

  /**
    * Write captured profile events in chromium format.
    @return the name of the profile file.
    */
  std::string EndProfiling() override;

 protected:
  /**
    * Load an ONNX model.
    * @param protobuf object corresponding to the model file. model_proto will be copied by the API.
    * @return OK if success.
    */
  common::Status Load(const ONNX_NAMESPACE::ModelProto& model_proto) override;

  /**
    * Load an ONNX model.
    * @param protobuf object corresponding to the model file. This is primarily to support large models.
    * @return OK if success.
    */
  common::Status Load(std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_model_proto) override;

 private:
  friend class Session;
  InferenceSession(const SessionOptions& session_options,
                   logging::LoggingManager* logging_manager);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(InferenceSession);

  bool HasLocalSchema() const {
    return !custom_schema_registries_.empty();
  }

  common::Status DoPostLoadProcessing(onnxruntime::Model& model);

  common::Status SaveModelMetadata(const onnxruntime::Model& model);

  // Create a Logger for a single execution if possible. Otherwise use the default logger.
  // If a new logger is created, it will also be stored in new_run_logger,
  // which must remain valid for the duration of the execution.
  // If the default logger is used, new_run_logger will remain empty.
  // The returned value should be used in the execution.
  const logging::Logger& CreateLoggerForRun(const RunOptions& run_options,
                                            std::unique_ptr<logging::Logger>& new_run_logger);

  common::Status Load(std::function<common::Status(std::shared_ptr<Model>&)> loader, const std::string& event_name);

  common::Status TransformGraph(onnxruntime::Graph& graph,
                                const onnxruntime::GraphTransformerManager& graph_transformer_mgr,
                                const ExecutionProviders& providers,
                                KernelRegistryManager& kernel_registry_manager,
                                const InsertCastTransformer& insert_cast_transformer,
                                SessionState& session_state);

  common::Status CreateSubgraphSessionState(Graph& graph, SessionState& session_state);

  common::Status InitializeSubgraphSessions(Graph& graph, SessionState& session_state);

  void AddPredefinedTransformers(GraphTransformerManager& transformer_manager,
                                 const uint32_t& graph_optimization_level,
                                 const std::vector<std::string>& custom_list);

  void InitLogger(logging::LoggingManager* logging_manager);

  static common::Status CheckTypes(MLDataType actual, MLDataType expected);

  common::Status ValidateInputs(const std::vector<std::string>& feed_names,
                                const std::vector<MLValue>& feeds);

  common::Status ValidateOutputs(const std::vector<std::string>& output_names,
                                 const std::vector<MLValue>* p_fetches);

  common::Status WaitForNotification(Notification* p_executor_done, int64_t timeout_in_ms);

  template <typename T>
  common::Status Load(const T& model_uri);

  template <typename T>
  void StartProfiling(const std::basic_string<T>& file_prefix);

  const SessionOptions session_options_;

  onnxruntime::GraphTransformerManager graph_transformation_mgr_;

  // List of transformers to run. When this list is not empty only the transformers in this list
  // will be run regardless of the level set.
  // .i.e This list overrides both SessionOptions.graph_optimization_level and predefined transformers.
  std::vector<std::string> transformers_to_enable_;

  /// Logging manager if provided.
  logging::LoggingManager* logging_manager_;

  /// Logger for this session. WARNING: Will contain nullptr if logging_manager_ is nullptr.
  std::unique_ptr<logging::Logger> owned_session_logger_;

  /// convenience pointer to logger. should always be the same as session_state_.Logger();
  const logging::Logger* session_logger_;

  // Profiler for this session.
  profiling::Profiler session_profiler_;

  ExecutionProviders execution_providers_;

  KernelRegistryManager kernel_registry_manager_;
  std::list<std::shared_ptr<onnxruntime::IOnnxRuntimeOpSchemaCollection>> custom_schema_registries_;

  // The model served by this inference session instance.
  // Currently this has to be a shared ptr because the Model::Load method
  // returns a shared_ptr only. Ideally factory functions should always return
  // unique_ptr for maximum flexibility. Client can always upgrade it to shared_ptr
  // if they need.
  std::shared_ptr<onnxruntime::Model> model_;

  // A set of executors that can run in parallel.
  std::vector<std::unique_ptr<IExecutor>> executors_;  // TODO do we need this vector?

  // Immutable state for each op in the model. Shared by all executors.
  SessionState session_state_;

  ModelMetadata model_metadata_;
  InputDefList required_input_def_list_;
  std::unordered_map<std::string, const NodeArg*> input_def_map_;
  OutputDefList output_def_list_;

  // names of model inputs and outputs used for quick validation.
  std::unordered_set<std::string> required_model_input_names_;
  std::unordered_set<std::string> model_input_names_;
  std::unordered_set<std::string> model_output_names_;

// Environment for this session
// not used now; we'll need it when we introduce threadpool
// statically allocated pointer, no need to manage its lifetime.
//Env* env_;

// Threadpool for this session
//thread::ThreadPool thread_pool_; // not used for now; will add it later when implementing RunAsync
#ifdef USE_EIGEN_THREADPOOL
  std::unique_ptr<Eigen::NonBlockingThreadPool> thread_pool_;
#else
  std::unique_ptr<TaskThreadPool> thread_pool_;
#endif

  // Number of concurrently running executors
  std::atomic<int> current_num_runs_;

  mutable onnxruntime::OrtMutex session_mutex_;  // to ensure only one thread can invoke Load/Initialize
  bool is_model_loaded_ = false;                 // GUARDED_BY(session_mutex_)
  bool is_inited_ = false;                       // GUARDED_BY(session_mutex_)

  InsertCastTransformer insert_cast_transformer_;
  // The file path of where the model was loaded. e.g. /tmp/test_squeezenet/model.onnx
  std::basic_string<PATH_CHAR_TYPE> model_location_;
};
}  // namespace onnxruntime

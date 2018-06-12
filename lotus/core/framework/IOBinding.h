#pragma once

#include "core/framework/execution_provider.h"
#include "core/common/status.h"
#include "core/framework/ml_value.h"
#include "core/framework/inference_session.h"
#include "core/common/logging/logging.h"

namespace Lotus {
/**
  * Input/Output binding.
  * Usage is as follows:
  *
  * InferenceSession session;
  * session.Load();
  * session.Initialize();
  * ...
  * shared_ptr<IOBinding> io_binding;
  * session.NewIOBinding("DML", &io_binding);
  * io_binding->BindInput(...);
  * io_binding->BindInput(...);
  * io_binding->SynchronizeInputs();
  *
  * io_binding->BindOutput(...);
  * io_binding->BindOutput(...);
  *
  * session.Run(io_binding);
  *
  * vector<MLValue>& outputs = io_binding->GetOutputs();
  */
class IOBinding {
 public:
  /**
    * Call repeatedly to bind as many inputs as required.
    * If the input mlvalue is not at the desired location (specified by the execution provider), this will
    * copy it to the desired location. This copy may or may not be async. It depends on the exec provider.
    * For copying it leverages IExecutionProvider::CopyTensor().
    */
  Common::Status BindInput(const std::string& name, const MLValue& ml_value);

  /**
    * If the BindInput calls are async this function acts as a barrier to ensure all inputs are fully copied
    * before you call the Run() method. There is no point calling Run() if you're inputs are not ready at the 
    * desired location.
    * This is a blocking call and is a wrapper over IExecutionProvider::Sync().
    * Call InferenceSession::Run() only after calling this method or else you'll end up wasting cycles inside Run().
    */
  Common::Status SynchronizeInputs();

  /**
    * This simply provides the names and optionally allocated output containers.
    */
  Common::Status BindOutput(const std::string& name, const MLValue& ml_value);

  /**
    * This simply collects the outputs obtained after calling Run() inside the @param outputs.
    */
  const std::vector<std::string>& GetOutputNames() const;
  std::vector<MLValue>& GetOutputs();

  const std::unordered_map<std::string, MLValue>& GetInputs() const;

 private:
  friend InferenceSession;

  IOBinding(IExecutionProvider* p_exec_provider, const Logging::Logger* p_logger);
  IExecutionProvider* p_exec_provider_ = nullptr;  // owned by session
  std::unordered_map<std::string, MLValue> feeds_;
  std::vector<std::string> output_names_;
  std::vector<MLValue> outputs_;
  const Logging::Logger* p_logger_ = nullptr;  // owned by session

  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(IOBinding);
};
}  // namespace Lotus

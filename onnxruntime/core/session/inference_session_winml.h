#pragma once

#include "core/common/status.h"
#include "core/session/inference_session.h"

namespace onnxruntime {

class InferenceSessionWinML : public InferenceSession {
 public:
  /**
    Create a new InferenceSessionWinML
    @param session_options Session options.
    @param logging_manager
    Optional logging manager instance that will enable per session logger output using
    session_options.session_logid as the logger id in messages.
    If nullptr, the default LoggingManager MUST have been created previously as it will be used
    for logging. This will use the default logger id in messages.
    See core/common/logging/logging.h for details on how to do that, and how LoggingManager::DefaultLogger works.
    */
  explicit InferenceSessionWinML(const SessionOptions& session_options,
                                 Logging::LoggingManager* logging_manager = nullptr)
      : InferenceSession(session_options, logging_manager) {}

  ~InferenceSessionWinML() = default;

  /**
    * Load an ONNX model.
    * @param protobuf object corresponding to the model file. model_proto will be copied by the API.
    * @return OK if success.
    */
  common::Status Load(const ONNX_NAMESPACE::ModelProto& model_proto);

  /**
    * Load an ONNX model.
    * @param protobuf object corresponding to the model file. This is primarily supported to support large models.
    * @return OK if success.
    */
  common::Status Load(std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_model_proto);

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(InferenceSessionWinML);
};

}  // namespace onnxruntime

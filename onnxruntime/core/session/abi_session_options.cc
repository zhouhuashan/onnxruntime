// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include <cstring>
#include "core/session/inference_session.h"
#include "abi_session_options_impl.h"

ONNX_RUNTIME_EXPORT ONNXRuntimeSessionOptions* CreateONNXRuntimeSessionOptions() NO_EXCEPTION {
  std::unique_ptr<ONNXRuntimeSessionOptions> options = std::make_unique<ONNXRuntimeSessionOptions>();
#ifdef USE_CUDA
  options->enable_cuda_provider = true;
#else
  options->enable_cuda_provider = false;
#endif
#ifdef USE_MKLDNN
  options->enable_mkl_provider = true;
#else
  options->enable_mkl_provider = false;
#endif
  return options.release();
}

ONNX_RUNTIME_EXPORT ONNXRuntimeSessionOptions* CloneONNXRuntimeSessionOptions(ONNXRuntimeSessionOptions* input) NO_EXCEPTION {
  try {
    return new ONNXRuntimeSessionOptions(*input);
  } catch (std::exception&) {
    return nullptr;
  }
}

ONNX_RUNTIME_EXPORT void ReleaseONNXRuntimeSessionOptions(_Frees_ptr_opt_ ONNXRuntimeSessionOptions* value) NO_EXCEPTION {
  delete value;
}

ONNX_RUNTIME_EXPORT void ONNXRuntimeEnableSequentialExecution(_In_ ONNXRuntimeSessionOptions* options) NO_EXCEPTION {
  options->value.enable_sequential_execution = true;
}
ONNX_RUNTIME_EXPORT void ONNXRuntimeDisableSequentialExecution(_In_ ONNXRuntimeSessionOptions* options) NO_EXCEPTION {
  options->value.enable_sequential_execution = false;
}

// enable profiling for this session.
ONNX_RUNTIME_EXPORT void ONNXRuntimeEnableProfiling(_In_ ONNXRuntimeSessionOptions* options, _In_ const char* profile_file_prefix) NO_EXCEPTION {
  options->value.enable_profiling = true;
  options->value.profile_file_prefix = profile_file_prefix;
}
ONNX_RUNTIME_EXPORT void ONNXRuntimeDisableProfiling(_In_ ONNXRuntimeSessionOptions* options) NO_EXCEPTION {
  options->value.enable_profiling = false;
  options->value.profile_file_prefix.clear();
}

// enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
ONNX_RUNTIME_EXPORT void ONNXRuntimeEnableMemPattern(_In_ ONNXRuntimeSessionOptions* options) NO_EXCEPTION {
  options->value.enable_mem_pattern = true;
}
ONNX_RUNTIME_EXPORT void ONNXRuntimeDisableMemPattern(_In_ ONNXRuntimeSessionOptions* options) NO_EXCEPTION {
  options->value.enable_mem_pattern = false;
}

// enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ONNX_RUNTIME_EXPORT void ONNXRuntimeEnableCpuMemArena(_In_ ONNXRuntimeSessionOptions* options) NO_EXCEPTION {
  options->value.enable_cpu_mem_arena = true;
}

ONNX_RUNTIME_EXPORT void ONNXRuntimeDisableCpuMemArena(_In_ ONNXRuntimeSessionOptions* options) NO_EXCEPTION {
  options->value.enable_cpu_mem_arena = false;
}

///< logger id to use for session output
ONNX_RUNTIME_EXPORT void ONNXRuntimeSetSessionLogId(_In_ ONNXRuntimeSessionOptions* options, const char* logid) NO_EXCEPTION {
  options->value.session_logid = logid;
}

///< applies to session load, initialization, etc
ONNX_RUNTIME_EXPORT void ONNXRuntimeSetSessionLogVerbosityLevel(_In_ ONNXRuntimeSessionOptions* options, uint32_t session_log_verbosity_level) NO_EXCEPTION {
  options->value.session_log_verbosity_level = session_log_verbosity_level;
}

///How many threads in the session thread pool.
ONNX_RUNTIME_EXPORT int ONNXRuntimeSetSessionThreadPoolSize(_In_ ONNXRuntimeSessionOptions* options, int session_thread_pool_size) NO_EXCEPTION {
  if (session_thread_pool_size <= 0) return -1;
  options->value.session_thread_pool_size = session_thread_pool_size;
  return 0;
}

ONNX_RUNTIME_EXPORT int ONNXRuntimeEnableCudaProvider(_In_ ONNXRuntimeSessionOptions* options, int device_id) NO_EXCEPTION {
#ifdef USE_CUDA
  options->enable_cuda_provider = true;
  options->cuda_device_id = device_id;
  return 0;
#else
  (void)options;
  (void)device_id;
  return -1;
#endif
}

ONNX_RUNTIME_EXPORT void ONNXRuntimeDisableCudaProvider(_In_ ONNXRuntimeSessionOptions* options) NO_EXCEPTION {
  options->enable_cuda_provider = false;
}

ONNX_RUNTIME_EXPORT int ONNXRuntimeEnableMklProvider(_In_ ONNXRuntimeSessionOptions* options) NO_EXCEPTION {
  //TODO:
  (void)options;
  return -1;
}

ONNX_RUNTIME_EXPORT void ONNXRuntimeDisableMklProvider(_In_ ONNXRuntimeSessionOptions* options) NO_EXCEPTION {
  options->enable_mkl_provider = false;
}

ONNX_RUNTIME_EXPORT void ONNXRuntimeAddCustomOp(_In_ ONNXRuntimeSessionOptions* options, const char* custom_op_path) NO_EXCEPTION {
  options->custom_op_paths.emplace_back(custom_op_path);
}
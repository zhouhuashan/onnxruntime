// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testenv.h"
#include <core/common/logging/logging.h>
#include <core/graph/constants.h>
#include <core/framework/allocator.h>
#include <core/providers/provider_factories.h>
#include <experimental/filesystem>
#ifdef _MSC_VER
#include <filesystem>
#endif

using namespace std::experimental::filesystem::v1;
using onnxruntime::Status;

Status SessionFactory::create(std::shared_ptr<::onnxruntime::InferenceSession>& sess, const path& model_url, const std::string& logid) const {
  ::onnxruntime::SessionOptions so;
  so.session_logid = logid;
  so.enable_cpu_mem_arena = enable_cpu_mem_arena_;
  so.enable_mem_pattern = enable_mem_pattern_;
  so.enable_sequential_execution = enable_sequential_execution;
  so.session_thread_pool_size = session_thread_pool_size;
  sess.reset(new ::onnxruntime::InferenceSession(so));

  Status status;
  for (const std::string& provider : providers_) {
    if (provider == onnxruntime::kCudaExecutionProvider) {
#ifdef USE_CUDA
      onnxruntime::CUDAExecutionProviderInfo cuda_pi;
      cuda_pi.device_id = 0;
      ONNXRUNTIME_RETURN_IF_ERROR(sess->RegisterExecutionProvider(::onnxruntime::CreateCUDAExecutionProvider(cuda_pi)));
#else
      ONNXRUNTIME_THROW("This executable was not built with CUDA");
#endif
    }
    //TODO: add more
  }
#ifdef USE_MKLDNN
  ::onnxruntime::CPUExecutionProviderInfo mkldnn_pi;
  mkldnn_pi.create_arena = enable_cpu_mem_arena_;
  ONNXRUNTIME_RETURN_IF_ERROR(sess->RegisterExecutionProvider(::onnxruntime::CreateMKLDNNExecutionProvider(mkldnn_pi)));
#endif
  ::onnxruntime::CPUExecutionProviderInfo cpu_pi;
  cpu_pi.create_arena = enable_cpu_mem_arena_;
  status = sess->RegisterExecutionProvider(::onnxruntime::CreateBasicCPUExecutionProvider(cpu_pi));

  status = sess->Load(model_url.string());
  ONNXRUNTIME_RETURN_IF_ERROR(status);
  LOGS_DEFAULT(INFO) << "successfully loaded model from " << model_url;
  status = sess->Initialize();
  if (status.IsOK())
    LOGS_DEFAULT(INFO) << "successfully initialized model from " << model_url;
  return status;
}

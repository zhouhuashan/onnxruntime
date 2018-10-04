// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testenv.h"
#include "FixedCountFinishCallback.h"
#include "test/util/include/default_providers.h"
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
TestEnv::TestEnv(const std::vector<ITestCase*>& tests1, TestResultStat& stat1, SessionFactory& sf1)
    : tests(tests1), next_test_to_run(0), stat(stat1), finished(new FixedCountFinishCallback(static_cast<int>(tests1.size()))), sf(sf1) {
}

TestEnv::~TestEnv() {
  delete finished;
}

Status SessionFactory::create(std::shared_ptr<::onnxruntime::InferenceSession>& sess, const path& model_url, const std::string& logid) const {
  ::onnxruntime::SessionOptions so;
  so.session_logid = logid;
  so.enable_cpu_mem_arena = enable_cpu_mem_arena_;
  so.enable_mem_pattern = enable_mem_pattern_;
  so.enable_sequential_execution = enable_sequential_execution;
  so.session_thread_pool_size = session_thread_pool_size;
  sess.reset(new ::onnxruntime::InferenceSession(so));

  Status status;
  for (const std::string& provider_type : providers_) {
    std::unique_ptr<::onnxruntime::IExecutionProvider> execution_provider;
    if (provider_type == ::onnxruntime::kCudaExecutionProvider)
      execution_provider = ::onnxruntime::test::DefaultCudaExecutionProvider();
    else if (provider_type == ::onnxruntime::kMklDnnExecutionProvider)
      execution_provider = ::onnxruntime::test::DefaultMkldnnExecutionProvider(enable_cpu_mem_arena_);
    else if (provider_type == ::onnxruntime::kCpuExecutionProvider)
      execution_provider = ::onnxruntime::test::DefaultCpuExecutionProvider(enable_cpu_mem_arena_);
    else if (provider_type == ::onnxruntime::kNupharExecutionProvider)
      execution_provider = ::onnxruntime::test::DefaultNupharExecutionProvider();

    if (execution_provider == nullptr)
      ONNXRUNTIME_THROW("This executable was not built with ", provider_type);
    ONNXRUNTIME_RETURN_IF_ERROR(sess->RegisterExecutionProvider(std::move(execution_provider)));
  }

  status = sess->Load(model_url.string());
  ONNXRUNTIME_RETURN_IF_ERROR(status);
  LOGS_DEFAULT(INFO) << "successfully loaded model from " << model_url;
  status = sess->Initialize();
  if (status.IsOK())
    LOGS_DEFAULT(INFO) << "successfully initialized model from " << model_url;
  return status;
}

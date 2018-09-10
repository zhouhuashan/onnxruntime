#ifdef USE_CUDA
#include <core/providers/cuda/cuda_execution_provider.h>  //TODO(@chasun): this is a temp hack
#endif
#ifdef USE_MKLDNN
#include <core/providers/mkldnn/mkldnn_execution_provider.h>
#endif
#include "testenv.h"

#include "FixedCountFinishCallback.h"

#include <core/graph/constants.h>
#include <core/framework/allocator.h>
#include <core/common/logging/logging.h>

#include <experimental/filesystem>
#ifdef _MSC_VER
#include <filesystem>
#endif

using namespace std::experimental::filesystem::v1;

TestEnv::TestEnv(const std::vector<ITestCase*>& tests1, TestResultStat& stat1, SessionFactory& sf1)
    : next_test_to_run(0), tests(tests1), stat(stat1), finished(new FixedCountFinishCallback(static_cast<int>(tests1.size()))), sf(sf1) {
}

TestEnv::~TestEnv() {
  delete finished;
}

::onnxruntime::common::Status SessionFactory::create(std::shared_ptr<::onnxruntime::InferenceSession>& sess, const path& model_url, const std::string& logid) const {
  ::onnxruntime::SessionOptions so;
  so.session_logid = logid;
  so.enable_cpu_mem_arena = enable_cpu_mem_arena_;
  so.enable_mem_pattern = enable_mem_pattern_;
  so.enable_sequential_execution = enable_sequential_execution;
  so.session_thread_pool_size = session_thread_pool_size;
  sess.reset(new ::onnxruntime::InferenceSession(so));

  ::onnxruntime::common::Status status;
  if (provider_ == onnxruntime::kCudaExecutionProvider) {
#if USE_CUDA
    ::onnxruntime::CUDAExecutionProviderInfo cuda_epi;
    cuda_epi.device_id = 0;
    status = sess->RegisterExecutionProvider(std::make_unique<::onnxruntime::CUDAExecutionProvider>(cuda_epi));
    LOTUS_RETURN_IF_ERROR(status);
#else
    LOTUS_THROW("This executable was not built with CUDA");
#endif
  }

  if (provider_ == onnxruntime::kMklDnnExecutionProvider) {
#if USE_MKLDNN
    onnxruntime::MKLDNNExecutionProviderInfo mkldnn_epi;
    status = sess->RegisterExecutionProvider(std::make_unique<onnxruntime::MKLDNNExecutionProvider>(mkldnn_epi));
    LOTUS_RETURN_IF_ERROR(status);
#else
    LOTUS_THROW("This executable was not built with MKLDNN");
#endif
  }

  status = sess->Load(model_url.string());
  LOTUS_RETURN_IF_ERROR(status);
  LOGS_DEFAULT(INFO) << "successfully loaded model from " << model_url;
  status = sess->Initialize();
  if (status.IsOK())
    LOGS_DEFAULT(INFO) << "successfully initialized model from " << model_url;
  return status;
}

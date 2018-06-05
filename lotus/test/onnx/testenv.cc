#include "testenv.h"
#include "FixedCountFinishCallback.h"
#include <core/graph/constants.h>
#include <core/framework/allocator.h>
#ifdef USE_CUDA
#include <core/providers/cuda/cuda_execution_provider.h>
#endif

TestEnv::TestEnv(const std::vector<ITestCase*>& tests1, TestResultStat& stat1, SessionFactory& sf1)
    : next_test_to_run(0), tests(tests1), stat(stat1), finished(new FixedCountFinishCallback((int)tests1.size())), sf(sf1) {
}

Lotus::Common::Status SessionFactory::create(std::shared_ptr<Lotus::InferenceSession>& sess, const std::string& model_url, const std::string& logid) const {
  Lotus::SessionOptions so;
  so.session_logid = logid;
  sess.reset(new Lotus::InferenceSession(so));
  Lotus::Common::Status status;
  if (provider == LotusIR::kCudaExecutionProvider) {
#if USE_CUDA
    Lotus::CUDAExecutionProviderInfo cuda_epi;
    cuda_epi.device_id = 0;
    status = session_object->RegisterExecutionProvider(std::make_unique<Lotus::CUDAExecutionProvider>(cuda_epi));
    LOTUS_RETURN_IF_ERROR(status);
#else
    LOTUS_THROW("This executable is not built with CUDA");
#endif
  }
  status = sess->Load(model_url);
  LOTUS_RETURN_IF_ERROR(status);
  LOGS_DEFAULT(INFO) << "successfully loaded model from " << model_url;
  status = sess->Initialize();
  if (status.IsOK())
    LOGS_DEFAULT(INFO) << "successfully initialized model from " << model_url;
  return status;
}
#include "sync_api.h"
#include <mutex>
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <core/common/common.h>
#include <core/common/logging/logging.h>
#include "simple_thread_pool.h"
#include "lotus_event.h"

using Lotus::Common::Status;

//this can be passed to one of the following functions:
//LotusSetEventWhenCallbackReturns
class LotusCallbackInstance {
 private:
  std::vector<LOTUS_EVENT> events_to_signal_;

 public:
  void AddEvent(LOTUS_EVENT event);
  Lotus::Common::Status SignalAllEvents();
};

Status WaitAndCloseEvent(LOTUS_EVENT finish_event) {
  if (finish_event == nullptr)
    return Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_ARGUMENT, "");
  pthread_mutex_lock(&finish_event->finish_event_mutex);
  while (!finish_event->finished) {
    pthread_cond_wait(&finish_event->finish_event_data, &finish_event->finish_event_mutex);
  }
  pthread_mutex_unlock(&finish_event->finish_event_mutex);
  delete finish_event;
  return Status::OK();
}

Status CreateAndSubmitThreadpoolWork(LOTUS_CALLBACK_FUNCTION callback, void* data, PThreadPool pool) {
  if (callback == nullptr)
    return Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_ARGUMENT, "callback cannot be NULL");
  if (pool == nullptr)
    return Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_ARGUMENT, "pool cannot be NULL");
  pool->Schedule([=]() {
    LotusCallbackInstance instance;
    callback(&instance, data, nullptr);
    Status st = instance.SignalAllEvents();
    if (!st.IsOK()) {
      LOGF_DEFAULT(ERROR, "SignalAllEvents failed:%s. aborting...\n", st.ErrorMessage().c_str());
      abort();
    }
  });
  return Status::OK();
}

using DefaultThreadPoolType = Lotus::SimpleThreadPoolTempl<Lotus::Env>;
static std::unique_ptr<DefaultThreadPoolType> default_pool;
static std::once_flag default_pool_init;

PThreadPool GetDefaultThreadPool(const Lotus::Env& env) {
  std::call_once(default_pool_init, [&env] {
    int core_num = env.GetNumCpuCores();
    default_pool.reset(new DefaultThreadPoolType(core_num, env));
  });
  return default_pool.get();
}

Status LotusSetEventWhenCallbackReturns(LOTUS_CALLBACK_INSTANCE pci, LOTUS_EVENT finish_event) {
  if (finish_event == nullptr)
    return Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_ARGUMENT, "");

  if (pci == nullptr) {
    if (pthread_mutex_lock(&finish_event->finish_event_mutex)) {
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "lock failed");
    }
    finish_event->finished = true;
    if (pthread_mutex_unlock(&finish_event->finish_event_mutex))
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "unlock failed");
    if (!pthread_cond_broadcast(&finish_event->finish_event_data))
      return Status::OK();
    else
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "pthread_cond_broadcast failed");
  } else {
    pci->AddEvent(finish_event);
    return Status::OK();
  }
}

void LotusCallbackInstance::AddEvent(LOTUS_EVENT event) {
  events_to_signal_.push_back(event);
}

Status LotusCallbackInstance::SignalAllEvents() {
  for (LOTUS_EVENT finish_event : events_to_signal_) {
    if (pthread_mutex_lock(&finish_event->finish_event_mutex)) {
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "lock failed");
    }
    finish_event->finished = true;
    if (pthread_mutex_unlock(&finish_event->finish_event_mutex))
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "unlock failed");
    if (pthread_cond_broadcast(&finish_event->finish_event_data))
      return LOTUS_MAKE_STATUS(LOTUS, FAIL, "pthread_cond_broadcast failed");
  }
  return Status::OK();
}

Status CreateLotusEvent(LOTUS_EVENT* out) {
  if (out == nullptr)
    return Status(Lotus::Common::LOTUS, Lotus::Common::INVALID_ARGUMENT, "");
  *out = new LotusEvent();
  return Status::OK();
}

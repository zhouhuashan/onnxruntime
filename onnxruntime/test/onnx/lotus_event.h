#include <mutex>
#include <pthread.h>
#include <core/common/common.h>

struct LotusEvent {
 public:
  pthread_mutex_t finish_event_mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t finish_event_data = PTHREAD_COND_INITIALIZER;
  bool finished = false;
  LotusEvent() = default;

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(LotusEvent);
};

using LOTUS_EVENT = LotusEvent*;
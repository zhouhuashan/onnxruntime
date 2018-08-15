#pragma once

#include "sync_api.h"

#include <mutex>

template <typename T>
class FixedCountFinishCallbackImpl {
 private:
  //remain tasks
  int s_;
  std::mutex m_;
  LOTUS_EVENT finish_event_;
  bool failed = false;
  std::vector<std::shared_ptr<T>> results_;

 public:
  const std::vector<std::shared_ptr<T>>& getResults() const {
    return results_;
  }
  FixedCountFinishCallbackImpl(int s) : s_(s), results_(s) {
    LOTUS_ENFORCE(CreateLotusEvent(&finish_event_).IsOK());
  }

  Lotus::Common::Status fail(LOTUS_CALLBACK_INSTANCE pci) {
    {
      std::lock_guard<std::mutex> g(m_);
      failed = true;
      s_ = 0;  //fail earlier
    }
    return LotusSetEventWhenCallbackReturns(pci, finish_event_);
  }

  Lotus::Common::Status onFinished(size_t task_index, std::shared_ptr<T> result, LOTUS_CALLBACK_INSTANCE pci) {
    int v;
    {
      std::lock_guard<std::mutex> g(m_);
      v = --s_;
      results_.at(task_index) = result;
    }
    if (v == 0) {
      return LotusSetEventWhenCallbackReturns(pci, finish_event_);
    }
    return Lotus::Common::Status::OK();
  }

  bool shouldStop() {
    std::lock_guard<std::mutex> g(m_);
    return failed;
  }
  //this function can only be invoked once
  bool wait() {
    LOTUS_ENFORCE(WaitAndCloseEvent(finish_event_).IsOK());
    {
      std::lock_guard<std::mutex> g(m_);
      return !failed;
    }
  }
};

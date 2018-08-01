#pragma once

#include <condition_variable>
#include <mutex>
#include "TestCaseResult.h"
#include <Windows.h>

extern Lotus::Common::Status SetWindowsEvent(PTP_CALLBACK_INSTANCE pci, HANDLE finish_event);
template <typename T>
class FixedCountFinishCallbackImpl {
 private:
  //remain tasks
  int s_;
  std::mutex m_;
  HANDLE finish_event_;
  bool failed = false;
  std::vector<std::shared_ptr<T>> results_;

 public:
  const std::vector<std::shared_ptr<T>>& getResults() const {
    return results_;
  }
  FixedCountFinishCallbackImpl(int s) : s_(s), results_(s) {
    finish_event_ = CreateEvent(
        NULL,                // default security attributes
        TRUE,                // manual-reset event
        FALSE,               // initial state is nonsignaled
        NULL
    );
    if (finish_event_ == nullptr) {
      throw std::runtime_error("init failed");
    }
  }

  Lotus::Common::Status fail(PTP_CALLBACK_INSTANCE pci) {
    {
      std::lock_guard<std::mutex> g(m_);
      failed = true;
      s_ = 0;  //fail earlier
    }
    return SetWindowsEvent(pci, finish_event_);
  }

  Lotus::Common::Status onFinished(size_t task_index, std::shared_ptr<T> result, PTP_CALLBACK_INSTANCE pci) {
    int v;
    {
      std::lock_guard<std::mutex> g(m_);
      v = --s_;
      results_.at(task_index) = result;
    }
    if (v == 0) {
      return SetWindowsEvent(pci, finish_event_);
    }
    return Lotus::Common::Status::OK();
  }

  bool shouldStop() { return failed; }
  bool wait() {
    while (s_ > 0) {
      DWORD dwWaitResult = WaitForSingleObject(
          finish_event_,  // event handle
          INFINITE);      // indefinite wait
      if (dwWaitResult != WAIT_OBJECT_0)
        return false;
    }
    return !failed;
  }
};

typedef FixedCountFinishCallbackImpl<TestCaseResult> FixedCountFinishCallback;

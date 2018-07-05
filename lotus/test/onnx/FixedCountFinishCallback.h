#pragma once

#include <condition_variable>
#include <mutex>
#include "TestCaseResult.h"

template <typename T>
class FixedCountFinishCallbackImpl {
 private:
  //remain tasks
  int s_;
  std::mutex m_;
  std::condition_variable cond_;
  bool failed = false;
  std::vector<std::shared_ptr<T>> results_;

 public:
  const std::vector<std::shared_ptr<T>>& getResults() const {
    return results_;
  }
  FixedCountFinishCallbackImpl(int s) : s_(s), results_(s) {}

  void fail() {
    std::lock_guard<std::mutex> g(m_);
    s_ = 0;  //fail earlier
    failed = true;
    cond_.notify_all();
  }

  void onFinished(size_t task_index, std::shared_ptr<T> result) {
    std::lock_guard<std::mutex> g(m_);
    s_--;
    results_.at(task_index) = result;
    cond_.notify_all();
  }

  bool shouldStop() { return failed; }
  bool wait() {
    std::unique_lock<std::mutex> lk(m_);
    while (s_) {
      cond_.wait(lk);
    }
    return !failed;
  }

  template <typename T1>
  bool wait_for(const T1& dur) {
    std::unique_lock<std::mutex> lk(m_);
    while (s_) {
      std::cv_status state = cond_.wait_for(lk, dur);
      if (state == std::cv_status::timeout) {
        return false;
      }
    }
    return !failed;
  }
};

using FixedCountFinishCallback = FixedCountFinishCallbackImpl<TestCaseResult>;

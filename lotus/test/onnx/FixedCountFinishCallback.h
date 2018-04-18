#pragma once

#include <condition_variable>
#include <mutex>
#include "IFinishCallback.h"

class FixedCountFinishCallback : public IFinishCallback {
 private:
  int s_;
  std::mutex m_;
  std::condition_variable cond_;
  bool failed = false;

 public:
  FixedCountFinishCallback(int s) : s_(s) {}
  virtual void onFinished(int retval) override {
    std::lock_guard<std::mutex> g(m_);
    if (retval == 0)
      s_--;
    else {
      s_ = 0;
      failed = true;
    }
    cond_.notify_all();
  }

  virtual bool shouldStop() override { return failed; }
  bool wait() override {
    std::unique_lock<std::mutex> lk(m_);
    while (s_) {
      cond_.wait(lk);
    }
    return !failed;
  }

  template <typename T>
  bool wait_for(const T& dur) {
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

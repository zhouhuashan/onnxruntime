/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/lib/threadpool.h"

// NonBlockingThreadPool.h(281): warning C4389: '==': signed/unsigned mismatch
// NonBlockingThreadPool.h(252): warning C4267: '-=': conversion from 'size_t' to 'unsigned int', possible loss of data
#pragma warning(disable : 4389 4267)
#include "core/common/logging/logging.h"
#include "core/platform/context.h"
#include "unsupported/Eigen/CXX11/ThreadPool"

namespace Lotus {
namespace thread {

struct EigenEnvironment {
  typedef Thread EnvThread;
  struct TaskImpl {
    std::function<void()> f;
    Context context;
    uint64 trace_id;
  };
  struct Task {
    std::unique_ptr<TaskImpl> f;
  };

  Env* const env_;
  const ThreadOptions thread_options_;
  const std::string name_;

  EigenEnvironment(Env* env, const ThreadOptions& thread_options,
                   const std::string& name)
      : env_(env), thread_options_(thread_options), name_(name) {}

  EnvThread* CreateThread(std::function<void()> f) {
    return env_->StartThread(thread_options_, name_, [=]() {
      // TODO
      // Set the processor flag to flush denormals to zero.
      // port::ScopedFlushDenormal flush;
      // TODO
      // Set the processor rounding mode to ROUND TO NEAREST.
      // port::ScopedSetRound round(FE_TONEAREST);
      f();
    });
  }

  Task CreateTask(std::function<void()> f) {
    uint64 id = 0;
    // TODO
    // if (port::Tracing::IsActive()) {
    //   id = port::Tracing::UniqueId();
    //   port::Tracing::RecordEvent(port::Tracing::EventCategory::kScheduleClosure,
    //                              id);
    // }
    return Task{
        std::unique_ptr<TaskImpl>(new TaskImpl{
            std::move(f),
            Context(ContextKind::kThread),
            id,
        }),
    };
  }

  void ExecuteTask(const Task& t) {
    WithContext wc(t.f->context);
    if (t.f->trace_id != 0) {
      // TODO
      // port::Tracing::ScopedActivity region(
      //     port::Tracing::EventCategory::kRunClosure, t.f->trace_id);
      t.f->f();
    } else {
      t.f->f();
    }
  }
};

struct ThreadPool::Impl : Eigen::ThreadPoolTempl<EigenEnvironment> {
  Impl(Env* env, const ThreadOptions& thread_options, const std::string& name,
       int num_threads, bool low_latency_hint)
      : Eigen::ThreadPoolTempl<EigenEnvironment>(
            num_threads, low_latency_hint,
            EigenEnvironment(env, thread_options, name)) {}
};

ThreadPool::ThreadPool(Env* env, const std::string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads, true) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const std::string& name, int num_threads)
    : ThreadPool(env, thread_options, name, num_threads, true) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const std::string& name, int num_threads,
                       bool low_latency_hint) {
  LOTUS_ENFORCE(num_threads >= 1);
  impl_.reset(new ThreadPool::Impl(env, thread_options, "lotus_" + name,
                                   num_threads, low_latency_hint));
}

ThreadPool::~ThreadPool() {}

void ThreadPool::Schedule(std::function<void()> fn) {
  LOTUS_ENFORCE(fn != nullptr);
  impl_->Schedule(std::move(fn));
}

int ThreadPool::NumThreads() const { return impl_->NumThreads(); }

int ThreadPool::CurrentThreadId() const { return impl_->CurrentThreadId(); }

}  // namespace thread
}  // namespace Lotus

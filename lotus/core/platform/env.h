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

#ifndef LOTUS_CORE_PLATFORM_ENV_H_
#define LOTUS_CORE_PLATFORM_ENV_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

#include "core/platform/env_time.h"
#include "core/platform/macros.h"
#include "core/platform/types.h"

namespace Lotus {

class Thread;
struct ThreadOptions;

/// \brief An interface used by the Lotus implementation to
/// access operating system functionality like the filesystem etc.
///
/// Callers may wish to provide a custom Env object to get fine grain
/// control.
///
/// All Env implementations are safe for concurrent access from
/// multiple threads without any external synchronization.
class Env {
 public:
  Env();
  virtual ~Env() = default;

  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static Env* Default();

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64 NowMicros() { return envTime->NowMicros(); };

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64 NowSeconds() { return envTime->NowSeconds(); }

  /// Sleeps/delays the thread for the prescribed number of micro-seconds.
  virtual void SleepForMicroseconds(int64 micros) = 0;
  
  /// \brief Returns a new thread that is running fn() and is identified
  /// (for debugging/performance-analysis) by "name".
  ///
  /// Caller takes ownership of the result and must delete it eventually
  /// (the deletion will block until fn() stops running).
  virtual Thread* StartThread(const ThreadOptions& thread_options,
                              const std::string& name,
                              std::function<void()> fn) = 0;


  // TODO add filesystem related functions
  
 private:
  LOTUS_DISALLOW_COPY_AND_ASSIGN(Env);
  EnvTime* envTime = EnvTime::Default();
};

/// Represents a thread used to run a Lotus function.
class Thread {
 public:
  Thread() {}

  /// Blocks until the thread of control stops running.
  virtual ~Thread();

 private:
  LOTUS_DISALLOW_COPY_AND_ASSIGN(Thread);
};

/// \brief Options to configure a Thread.
///
/// Note that the options are all hints, and the
/// underlying implementation may choose to ignore it.
struct ThreadOptions {
  /// Thread stack size to use (in bytes).
  size_t stack_size = 0;  // 0: use system default value
  /// Guard area size to use near thread stacks to use (in bytes)
  size_t guard_size = 0;  // 0: use system default value
};

}  // namespace Lotus

#endif  // LOTUS_CORE_PLATFORM_ENV_H_

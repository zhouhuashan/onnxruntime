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

#include <Shlwapi.h>
#include <Windows.h>

#include "core/common/logging.h"
#include "core/platform/env.h"
#include "core/platform/types.h"

#include <string>
#include <thread>

#pragma comment(lib, "Shlwapi.lib")

namespace Lotus {

namespace {

class StdThread : public Thread {
 public:
  // name and thread_options are both ignored for now.
  StdThread(const ThreadOptions&, const std::string&,
            std::function<void()> fn)
      : thread_(fn) {}
  ~StdThread() { thread_.join(); }

 private:
  std::thread thread_;
};

class WindowsEnv : public Env {
 public:
  WindowsEnv()
      : GetSystemTimePreciseAsFileTime_(nullptr) {
    // GetSystemTimePreciseAsFileTime function is only available in the latest
    // versions of Windows. For that reason, we try to look it up in
    // kernel32.dll at runtime and use an alternative option if the function
    // is not available.
    HMODULE module = GetModuleHandleW(L"kernel32.dll");
    if (module != nullptr) {
      auto func = (FnGetSystemTimePreciseAsFileTime)GetProcAddress(
          module, "GetSystemTimePreciseAsFileTime");
      GetSystemTimePreciseAsFileTime_ = func;
    }
  }

  ~WindowsEnv() override {
    LOG(FATAL) << "Env::Default() must not be destroyed";
  }

  void SleepForMicroseconds(int64 micros) override { Sleep(static_cast<DWORD>(micros) / 1000); }

  Thread* StartThread(const ThreadOptions& thread_options, const std::string& name,
                      std::function<void()> fn) override {
    return new StdThread(thread_options, name, fn);
  }

 private:
  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  FnGetSystemTimePreciseAsFileTime GetSystemTimePreciseAsFileTime_;
};

}  // namespace

#if defined(PLATFORM_WINDOWS)
Env* Env::Default() {
  static Env* default_env = new WindowsEnv;
  return default_env;
}
#endif

}  // namespace Lotus

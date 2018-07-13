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
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <thread>
#include <vector>

#include "core/platform/env.h"
#include "core/platform/types.h"
#include "core/common/common.h"

namespace Lotus {

namespace {

class StdThread : public Thread {
 public:
  // name and thread_options are both ignored.
  StdThread(const ThreadOptions& thread_options, const std::string& name,
            std::function<void()> fn)
      : thread_(fn) {}
  ~StdThread() override { thread_.join(); }

 private:
  std::thread thread_;
};

class PosixEnv : public Env {
 public:
  static PosixEnv& Instance() {
    static PosixEnv default_env;
    return default_env;
  }

  int GetNumCpuCores() const override {
    // TODO if you need the number of physical cores you'll need to parse
    // /proc/cpuinfo and grep for "cpu cores".
    //However, that information is not always available(output of 'grep -i core /proc/cpuinfo' is empty)
    return std::thread::hardware_concurrency();
  }

  void SleepForMicroseconds(int64 micros) const override {
    while (micros > 0) {
      timespec sleep_time;
      sleep_time.tv_sec = 0;
      sleep_time.tv_nsec = 0;

      if (micros >= 1e6) {
        sleep_time.tv_sec =
            std::min<int64>(micros / 1e6, std::numeric_limits<time_t>::max());
        micros -= static_cast<int64>(sleep_time.tv_sec) * 1e6;
      }
      if (micros < 1e6) {
        sleep_time.tv_nsec = 1000 * micros;
        micros = 0;
      }
      while (nanosleep(&sleep_time, &sleep_time) != 0 && errno == EINTR) {
        // Ignore signals and wait for the full interval to elapse.
      }
    }
  }

  Thread* StartThread(const ThreadOptions& thread_options, const std::string& name,
                      std::function<void()> fn) const override {
    return new StdThread(thread_options, name, fn);
  }
  Common::Status FileExists(const char* fname) const override {
    return Common::Status(Common::LOTUS, Common::NOT_IMPLEMENTED, "NOT_IMPLEMENTED");
  }
  Common::Status ReadFileAsString(const char* fname, std::string* out) const override {
    if (!out) {
      return Common::Status(Common::LOTUS, Common::INVALID_ARGUMENT, "'out' cannot be NULL");
    }
    char errbuf[512];
    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
      snprintf(errbuf, sizeof(errbuf), "%s:%d open file %s fail, errcode = %d", __FILE__, (int)__LINE__, fname, errno);
      return Common::Status(Common::LOTUS, Common::FAIL, errbuf);
    }
    struct stat stbuf;
    if ((fstat(fd, &stbuf) != 0) || (!S_ISREG(stbuf.st_mode))) {
      close(fd);
      snprintf(errbuf, sizeof(errbuf), "%s:%d read file %s fail", __FILE__, (int)__LINE__, fname);
      return Common::Status(Common::LOTUS, Common::FAIL, errbuf);
    }
    if (stbuf.st_size == 0) {
      out->clear();
    } else {
      out->resize(stbuf.st_size, '\0');
      ssize_t bytes_readed = read(fd, (void*)out->data(), stbuf.st_size);
      if (bytes_readed <= 0 || bytes_readed != stbuf.st_size) {
        close(fd);
        snprintf(errbuf,
                 sizeof(errbuf),
                 "%s:%d open file %s fail, errcode = %d",
                 __FILE__,
                 (int)__LINE__,
                 fname,
                 errno);
        return Common::Status(Common::LOTUS, Common::FAIL, errbuf);
      }
      close(fd);
    }
    return Common::Status::OK();
  }

 private:
  PosixEnv() = default;
};

}  // namespace

#if defined(PLATFORM_POSIX) || defined(__ANDROID__)
// REGISTER_FILE_SYSTEM("", PosixFileSystem);
// REGISTER_FILE_SYSTEM("file", LocalPosixFileSystem);
const Env& Env::Default() {
  return PosixEnv::Instance();
}
#endif

}  // namespace Lotus

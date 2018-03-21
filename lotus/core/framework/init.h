#ifndef CORE_FRAMEWORK_INIT_H
#define CORE_FRAMEWORK_INIT_H
#include "core/common/common.h"
#include "core/common/status.h"

namespace Lotus {
/**
    * Performs Lotus initialization logic at most once.
    */
class Initializer {
 public:
  Initializer(const Initializer&) = delete;
  Initializer& operator=(const Initializer&) = delete;

  /**
        * Runs the initialization logic if it hasn't been run yet.
        */
  static Status EnsureInitialized(int* pargc, char*** pargv);
  static Status EnsureInitialized();

 private:
  Initializer(int* pargc, char*** pargv);
  Status Initialize(int* pargc, char*** pargv);

  Status initialization_status_;
};
}  // namespace Lotus

#endif
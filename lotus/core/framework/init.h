#pragma once

#include "core/common/common.h"
#include "core/common/status.h"

namespace Lotus {
/**
    * Performs Lotus initialization logic at most once.
    */
class Initializer {
 public:
  /**
        * Runs the initialization logic if it hasn't been run yet.
        */
  static Status EnsureInitialized(int* pargc, char*** pargv);
  static Status EnsureInitialized();

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(Initializer);

  Initializer(int* pargc, char*** pargv);
  Status Initialize(int* pargc, char*** pargv);

  Status initialization_status_;
};
}  // namespace Lotus

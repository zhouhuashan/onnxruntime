#pragma once
#include "core/common/common.h"

class IFinishCallback {
 public:
  IFinishCallback() {}
  virtual ~IFinishCallback() {}
  virtual void onFinished(int retval) = 0;
  virtual bool shouldStop() = 0;
  virtual bool wait() = 0;

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(IFinishCallback);
};

#include "sync_api.h"
#include <core/common/common.h>

using ::Lotus::Common::Status;
Status CreateAndSubmitThreadpoolWork(LOTUS_CALLBACK_FUNCTION callback, void* data, PThreadPool pool) {
  PTP_WORK work = CreateThreadpoolWork(callback, data, pool);
  if (!work) {
    return Status(::Lotus::Common::LOTUS, ::Lotus::Common::FAIL, "create thread pool task failed");
  }
  SubmitThreadpoolWork(work);
  return Status::OK();
}

Status WaitAndCloseEvent(LOTUS_EVENT finish_event) {
  DWORD dwWaitResult = WaitForSingleObject(finish_event, INFINITE);
  (void)CloseHandle(finish_event);
  if (dwWaitResult != WAIT_OBJECT_0) {
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "WaitForSingleObject failed");
  }
  return Status::OK();
}

Status CreateLotusEvent(LOTUS_EVENT* out) {
  if (out == nullptr)
    return Status(::Lotus::Common::LOTUS, ::Lotus::Common::INVALID_ARGUMENT, "");
  HANDLE finish_event = CreateEvent(
      NULL,   // default security attributes
      TRUE,   // manual-reset event
      FALSE,  // initial state is nonsignaled
      NULL);
  if (finish_event == NULL) {
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "unable to create finish event");
  }
  *out = finish_event;
  return Status::OK();
}

Status LotusSetEventWhenCallbackReturns(LOTUS_CALLBACK_INSTANCE pci, LOTUS_EVENT finish_event) {
  if (finish_event == nullptr)
    return Status(::Lotus::Common::LOTUS, ::Lotus::Common::INVALID_ARGUMENT, "");
  if (pci)
    SetEventWhenCallbackReturns(pci, finish_event);
  else if (!SetEvent(finish_event)) {
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "SetEvent failed");
  }
  return Status::OK();
}

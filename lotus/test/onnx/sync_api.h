#pragma once

#ifdef _WIN32
#include <Windows.h>
#else
#include <vector>
#endif
#include <core/common/status.h>
#include <core/common/common.h>
#include <core/platform/env.h>

#ifdef _WIN32
using LOTUS_CALLBACK_INSTANCE = PTP_CALLBACK_INSTANCE;
using LOTUS_EVENT = HANDLE;
#define LOTUS_CALLBACK __stdcall
using LOTUS_WORK = PTP_WORK;
using PThreadPool = PTP_CALLBACK_ENVIRON;
using LOTUS_CALLBACK_FUNCTION = PTP_WORK_CALLBACK;
#define LotusCloseThreadpoolWork CloseThreadpoolWork
inline PThreadPool GetDefaultThreadPool(const ::Lotus::Env&) {
  return nullptr;
}
#else
#define LOTUS_CALLBACK
namespace Eigen {
class ThreadPoolInterface;
}
using PThreadPool = Eigen::ThreadPoolInterface*;
#define LOTUS_WORK void*
struct LotusEvent;
using LOTUS_EVENT = LotusEvent*;

class LotusCallbackInstance;
using LOTUS_CALLBACK_INSTANCE = LotusCallbackInstance*;
using LOTUS_CALLBACK_FUNCTION = void LOTUS_CALLBACK (*)(LOTUS_CALLBACK_INSTANCE pci, void* context, LOTUS_WORK work);
//Do nothing
inline void LotusCloseThreadpoolWork(LOTUS_WORK) {}
#endif

//The returned value will be used with CreateAndSubmitThreadpoolWork function
PThreadPool GetDefaultThreadPool(const ::Lotus::Env& env);
//On Windows, the last parameter can be null, in that case it will use the default thread pool.
//On Linux, there is no per process default thread pool. You have to pass a non-null pointer.
//Caller must delete the data pointer if this function returns a non-ok status. Otherwise, the ownership is transferred
::Lotus::Common::Status CreateAndSubmitThreadpoolWork(LOTUS_CALLBACK_FUNCTION callback, void* data, PThreadPool pool);
::Lotus::Common::Status CreateLotusEvent(LOTUS_EVENT* out);
//pci is a pointer, can be NULL. If pci is NULL, signal the event immediately
::Lotus::Common::Status LotusSetEventWhenCallbackReturns(LOTUS_CALLBACK_INSTANCE pci, LOTUS_EVENT finish_event);
::Lotus::Common::Status WaitAndCloseEvent(LOTUS_EVENT finish_event);

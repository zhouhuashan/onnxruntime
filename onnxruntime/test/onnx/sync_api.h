// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
using CALLBACK_INSTANCE = PTP_CALLBACK_INSTANCE;
using EVENT = HANDLE;
#define CALLBACK __stdcall
using WORK = PTP_WORK;
using PThreadPool = PTP_CALLBACK_ENVIRON;
using ONNXRUNTIME_CALLBACK_FUNCTION = PTP_WORK_CALLBACK;
#define OnnxRuntimeCloseThreadpoolWork CloseThreadpoolWork
inline PThreadPool GetDefaultThreadPool(const ::onnxruntime::Env&) {
  return nullptr;
}
#else
#define CALLBACK
namespace Eigen {
class ThreadPoolInterface;
}
using PThreadPool = Eigen::ThreadPoolInterface*;
#define WORK void*
struct OnnxRuntimeEvent;
using EVENT = OnnxRuntimeEvent*;

class OnnxRuntimeCallbackInstance;
using CALLBACK_INSTANCE = OnnxRuntimeCallbackInstance*;
using ONNXRUNTIME_CALLBACK_FUNCTION = void CALLBACK (*)(CALLBACK_INSTANCE pci, void* context, WORK work);
//Do nothing
inline void OnnxRuntimeCloseThreadpoolWork(WORK) {}
#endif

//The returned value will be used with CreateAndSubmitThreadpoolWork function
PThreadPool GetDefaultThreadPool(const ::onnxruntime::Env& env);
//On Windows, the last parameter can be null, in that case it will use the default thread pool.
//On Linux, there is no per process default thread pool. You have to pass a non-null pointer.
//Caller must delete the data pointer if this function returns a non-ok status. Otherwise, the ownership is transferred
::onnxruntime::common::Status CreateAndSubmitThreadpoolWork(ONNXRUNTIME_CALLBACK_FUNCTION callback, void* data, PThreadPool pool);
::onnxruntime::common::Status CreateOnnxRuntimeEvent(EVENT* out);
//pci is a pointer, can be NULL. If pci is NULL, signal the event immediately
::onnxruntime::common::Status OnnxRuntimeSetEventWhenCallbackReturns(CALLBACK_INSTANCE pci, EVENT finish_event);
::onnxruntime::common::Status WaitAndCloseEvent(EVENT finish_event);

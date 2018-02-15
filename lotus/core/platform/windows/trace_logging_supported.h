// @@COPYRIGHT@@
#ifndef _LOTUS_CORE_PLATFORM_WINDOWS_TRACE_LOGGING_ENABLED_H_
#define _LOTUS_CORE_PLATFORM_WINDOWS_TRACE_LOGGING_ENABLED_H_

#include <Windows.h>
#include <ntverp.h>

// check for Windows 10 SDK or later
// https://stackoverflow.com/questions/2665755/how-can-i-determine-the-version-of-the-windows-sdk-installed-on-my-computer
#if VER_PRODUCTBUILD > 9600
// LotusRT ETW trace logging uses Windows 10 SDK's TraceLoggingProvider.h
#define LOTUS_ETW_TRACE_LOGGING_SUPPORTED 1
#endif

#endif // _LOTUS_CORE_PLATFORM_WINDOWS_TRACE_LOGGING_ENABLED_H_

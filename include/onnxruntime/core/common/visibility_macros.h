// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
//define ONNX_RUNTIME_BUILD_DLL if your program is dynamically linked to onnxruntime
//define ONNX_RUNTIME_EXPORTS if it produces the onnxruntime.dll symbols
//DO NOT define ONNX_RUNTIME_EXPORTS if it consumes onnxruntime.dll
#ifdef _WIN32
#ifdef ONNX_RUNTIME_BUILD_DLL
#if ONNX_RUNTIME_EXPORTS
#define ONNX_RUNTIME_EXPORT __declspec(dllexport)
#else
#define ONNX_RUNTIME_EXPORT __declspec(dllimport)
#endif
#else
//building ONNX static libraries on Windows
#define ONNX_RUNTIME_EXPORT
#endif
#else
#define ONNX_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif

//SAL2 staffs
#ifndef _WIN32
#define _In_
#define _Out_
#define _Inout_
#define _Frees_ptr_opt_
#else
#include <specstrings.h>
#endif
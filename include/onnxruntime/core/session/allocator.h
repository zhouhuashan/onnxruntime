// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/error_code.h"
#include "core/framework/allocator_info.h"

typedef struct ONNXRuntimeAllocatorInteface {
  void* (*Alloc)(void* this_, size_t size);
  void (*Free)(void* this_, void* p);
  const ONNXRuntimeAllocatorInfo* (*Info)(void* this_);
  //These methods returns the new reference count.
  uint32_t (*AddRef)(void* this_);
  uint32_t (*Release)(void* this_);
} ONNXRuntimeAllocatorInteface;

typedef ONNXRuntimeAllocatorInteface* ONNXRuntimeAllocator;

#define ONNXRUNTIME_ALLOCATOR_IMPL_BEGIN(CLASS_NAME)               \
  class CLASS_NAME {                                               \
   private:                                                        \
    const ONNXRuntimeAllocatorInteface* vtable_ = &table_;         \
    static void* Alloc_(void* this_ptr, size_t size) {             \
      return ((CLASS_NAME*)this_ptr)->Alloc(size);                 \
    }                                                              \
    static void Free_(void* this_ptr, void* p) {                   \
      return ((CLASS_NAME*)this_ptr)->Free(p);                     \
    }                                                              \
    static const ONNXRuntimeAllocatorInfo* Info_(void* this_ptr) { \
      return ((CLASS_NAME*)this_ptr)->Info();                      \
    }                                                              \
    static uint32_t AddRef_(void* this_ptr) {                      \
      return ((CLASS_NAME*)this_ptr)->AddRef();                    \
    }                                                              \
    static uint32_t Release_(void* this_ptr) {                     \
      return ((CLASS_NAME*)this_ptr)->Release();                   \
    }                                                              \
    static constexpr ONNXRuntimeAllocatorInteface table_ = {       \
        Alloc_, Free_, Info_, AddRef_, Release_};

#define ONNXRUNTIME_ALLOCATOR_IMPL_END \
  }                                    \
  ;

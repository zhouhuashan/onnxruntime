// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "error_code.h"

typedef enum ONNXRuntimeAllocatorType {
  ONNXRuntimeMemDeviceAllocator = 0,
  ONNXRuntimeMemArenaAllocator = 1
} ONNXRuntimeAllocatorType;

// memory types for allocator, exec provider specific types should be extended in each provider
typedef enum ONNXRuntimeMemType {
  ONNXRuntimeMemTypeCPUInput = -2,                      // Any CPU memory used by non-CPU execution provider
  ONNXRuntimeMemTypeCPUOutput = -1,                     // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
  ONNXRuntimeMemTypeCPU = ONNXRuntimeMemTypeCPUOutput,  // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
  ONNXRuntimeMemTypeDefault = 0,                        // the default allocator for execution provider
} ONNXRuntimeMemType;

typedef struct ONNXRuntimeAllocatorInfo {
  const char* name;
  int id;
  enum ONNXRuntimeMemType mem_type;
  enum ONNXRuntimeAllocatorType type;
} ONNXRuntimeAllocatorInfo;

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

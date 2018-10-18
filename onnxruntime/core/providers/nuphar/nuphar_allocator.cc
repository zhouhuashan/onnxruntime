// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <tvm/runtime/device_api.h>

#include "nuphar_allocator.h"
#include "core/codegen/tvm/tvm_utils.h"

namespace onnxruntime {

void* NupharAllocator::Alloc(size_t size) {
  if (size == 0)
    return nullptr;

  // Alloc memory that meets the minimal alignment requirement imposed by TVM
  void *p = tvm::runtime::DeviceAPI::Get(tvm_ctx_)->AllocDataSpace(tvm_ctx_, size,
                                                                   /*alignment=*/tvm::runtime::kAllocAlignment,
                                                                   /*type_hint=*/{});
  if (p == nullptr) {
    ONNXRUNTIME_THROW("TVM AllocDataSpace failure");
  }
  return p;
}

void NupharAllocator::Free(void *p) {
  tvm::runtime::DeviceAPI::Get(tvm_ctx_)->FreeDataSpace(tvm_ctx_, p);
}

const ONNXRuntimeAllocatorInfo& NupharAllocator::Info() const {
  static ONNXRuntimeAllocatorInfo tvm_allocator_info(TVM_STACKVM, ONNXRuntimeAllocatorType::ONNXRuntimeDeviceAllocator,
                                                     tvm_ctx_.device_id, ONNXRuntimeMemTypeDefault);
  return tvm_allocator_info;
}

}  // namespace onnxruntime

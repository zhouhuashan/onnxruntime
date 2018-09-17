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
    LOTUS_THROW("TVM AllocDataSpace failure");
  }
  return p;
}

void NupharAllocator::Free(void *p) {
  tvm::runtime::DeviceAPI::Get(tvm_ctx_)->FreeDataSpace(tvm_ctx_, p);
}

const AllocatorInfo& NupharAllocator::Info() const {
  static AllocatorInfo tvm_allocator_info(TVM_STACKVM, AllocatorType::kDeviceAllocator,
                                          tvm_ctx_.device_id, kMemTypeDefault);
  return tvm_allocator_info;
}

}  // namespace onnxruntime

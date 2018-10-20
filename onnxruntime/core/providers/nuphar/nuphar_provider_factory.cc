// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/nuphar_provider_factory.h"
#include "nuphar_execution_provider.h"

using namespace onnxruntime;

namespace {
struct NupharProviderFactory {
  const ONNXRuntimeProviderFactoryInterface* const cls;
  std::atomic_int ref_count;
  NupharExecutionProviderInfo info;
  NupharProviderFactory();
};

ONNXStatusPtr ONNXRUNTIME_API_STATUSCALL CreateNuphar(void* this_, ONNXRuntimeProviderPtr* out) {
  NupharProviderFactory* this_ptr = (NupharProviderFactory*)this_;
  NupharExecutionProvider* ret = new NupharExecutionProvider(this_ptr->info);
  *out = (ONNXRuntimeProviderPtr)ret;
  return nullptr;
}

uint32_t ONNXRUNTIME_API_STATUSCALL ReleaseNuphar(void* this_) {
  NupharProviderFactory* this_ptr = (NupharProviderFactory*)this_;
  if (--this_ptr->ref_count == 0)
    delete this_ptr;
  return 0;
}

uint32_t ONNXRUNTIME_API_STATUSCALL AddRefNuphar(void* this_) {
  NupharProviderFactory* this_ptr = (NupharProviderFactory*)this_;
  ++this_ptr->ref_count;
  return 0;
}

constexpr ONNXRuntimeProviderFactoryInterface nuphar_cls = {
    AddRefNuphar,
    ReleaseNuphar,
    CreateNuphar,
};

NupharProviderFactory::NupharProviderFactory() : cls(&nuphar_cls), ref_count(1), info(0) {}
}  // namespace

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateNupharExecutionProviderFactory, int device_id, _In_ const char* target_str, _Out_ ONNXRuntimeProviderFactoryPtr** out) {
  NupharProviderFactory* ret = new NupharProviderFactory();
  ret->info.device_id = device_id;
  ret->info.target_str = target_str;
  *out = (ONNXRuntimeProviderFactoryPtr*)ret;
  return nullptr;
}

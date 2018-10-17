// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateCUDAExecutionProviderFactory, int device_id, _Out_ ONNXRuntimeProviderFactoryPtr** out);

#ifdef __cplusplus
}
#endif
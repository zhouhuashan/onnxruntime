// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include "core/session/inference_session.h"

struct ONNXRuntimeSessionOptions {
  onnxruntime::SessionOptions value;
  bool enable_cuda_provider;
  bool enable_mkl_provider;
  int cuda_device_id;
  std::vector<std::string> custom_op_paths;
};
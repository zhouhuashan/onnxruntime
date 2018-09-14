// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace ml {
#define ONNX_CONTRIB_OPERATOR_SCHEMA(name) \
  ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) \
  ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define ONNX_CONTRIB_OPERATOR_SCHEMA_UNIQ(Counter, name)         \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name##Counter) ONNX_UNUSED =      \
      ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__)

void RegisterContribSchemas();
void RegisterContribKernels(std::function<void(KernelCreateInfo&&)> create_fn);
}  // namespace ml
}  // namespace onnxruntime

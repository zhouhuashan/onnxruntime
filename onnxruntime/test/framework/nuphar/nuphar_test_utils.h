// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_TVM
#include "core/providers/nuphar/nuphar_execution_provider.h"
#endif  // USE_TVM

namespace onnxruntime {
namespace Test {
#ifdef USE_TVM
IExecutionProvider* TestNupharExecutionProvider();
#endif  // USE_TVM
}  // namespace Test
}  // namespace onnxruntime

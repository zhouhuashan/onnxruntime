// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nuphar_test_utils.h"

namespace onnxruntime {
namespace test {
#ifdef USE_TVM
IExecutionProvider* TestNupharExecutionProvider() {
  static NupharExecutionProviderInfo info(0);
  static NupharExecutionProvider nuphar_provider(info);
  return &nuphar_provider;
}
#endif  // USE_TVM
}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"
#include "core/framework/tensor.h"
#include "core/graph/onnx_protobuf.h"

#include "gtest/gtest.h"

#include <sstream>

namespace onnxruntime {
namespace Test {
#ifdef LOTUSIR_RUN_EXTERNAL_ONNX_TESTS
TEST(TensorProtoUtilsTest, test1) {
  const char* filename = "../models/test_resnet50/test_data_set_0/input_0.pb";
  std::string tensorbinary;
  common::Status st = Env::Default().ReadFileAsString(filename, &tensorbinary);
  ASSERT_TRUE(st.IsOK());
  ONNX_NAMESPACE::TensorProto proto;
  ASSERT_TRUE(proto.ParseFromString(tensorbinary));
  std::unique_ptr<Tensor> tensor;
  ::onnxruntime::AllocatorPtr cpu_allocator = std::make_shared<::onnxruntime::CPUAllocator>();
  st = ::onnxruntime::Utils::GetTensorFromTensorProto(proto, &tensor, cpu_allocator);
  ASSERT_TRUE(st.IsOK());
}
#endif
}  // namespace Test
}  // namespace onnxruntime

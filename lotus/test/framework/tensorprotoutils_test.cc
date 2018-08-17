#include "core/platform/env.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"

#include "gtest/gtest.h"

#include <sstream>

namespace Lotus {
namespace Test {
#ifdef LOTUSIR_RUN_EXTERNAL_ONNX_TESTS
TEST(TensorProtoUtilsTest, test1) {
  const char* filename = "../models/test_resnet50/test_data_set_0/input_0.pb";
  std::string tensorbinary;
  Common::Status st = Env::Default().ReadFileAsString(filename, &tensorbinary);
  ASSERT_TRUE(st.IsOK());
  onnx::TensorProto proto;
  ASSERT_TRUE(proto.ParseFromString(tensorbinary));
  std::unique_ptr<Tensor> tensor;
  ::Lotus::AllocatorPtr cpu_allocator = std::make_shared<::Lotus::CPUAllocator>();
  st = ::Lotus::Utils::GetTensorFromTensorProto(proto, &tensor, cpu_allocator);
  ASSERT_TRUE(st.IsOK());
}
#endif
}  // namespace Test
}  // namespace Lotus
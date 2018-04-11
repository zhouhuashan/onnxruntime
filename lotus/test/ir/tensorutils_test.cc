#include "core/graph/tensorutils.h"
#include "onnx/onnx-ml.pb.h"
#include "gtest/gtest.h"

using namespace Lotus::Utils;
using namespace onnx;

namespace Lotus {
namespace Test {
TEST(TensorParseTest, TensorUtilsTest) {
  TensorProto bool_tensor_proto;
  bool_tensor_proto.set_data_type(TensorProto_DataType_BOOL);
  bool_tensor_proto.add_int32_data(1);

  bool bool_data[1];
  auto status = TensorUtils::UnpackTensor(bool_tensor_proto, bool_data, 1);
  EXPECT_TRUE(status.IsOK());
  EXPECT_TRUE(bool_data[0]);

  float float_data[1];
  status = TensorUtils::UnpackTensor(bool_tensor_proto, float_data, 1);
  EXPECT_FALSE(status.IsOK());

  TensorProto float_tensor_proto;
  float_tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  float f[4] = {1.1f, 2.2f, 3.3f, 4.4f};
  char rawdata[sizeof(float) * 4 + 1];
  for (int i = 0; i < 4; ++i) {
    memcpy(rawdata + i * sizeof(float), &(f[i]), sizeof(float));
  }

  rawdata[sizeof(float) * 4] = '\0';
  float_tensor_proto.set_raw_data(rawdata);
  float float_data2[4];
  status = TensorUtils::UnpackTensor(float_tensor_proto, float_data2, 4);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(1.1f, float_data2[0]);
  EXPECT_EQ(2.2f, float_data2[1]);
  EXPECT_EQ(3.3f, float_data2[2]);
  EXPECT_EQ(4.4f, float_data2[3]);

  TensorProto string_tensor_proto;
  string_tensor_proto.set_data_type(TensorProto_DataType_STRING);
  string_tensor_proto.add_string_data("a");
  string_tensor_proto.add_string_data("b");

  std::string string_data[2];
  status = TensorUtils::UnpackTensor(string_tensor_proto, string_data, 2);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ("a", string_data[0]);
  EXPECT_EQ("b", string_data[1]);

  status = TensorUtils::UnpackTensor(bool_tensor_proto, string_data, 2);
  EXPECT_FALSE(status.IsOK());
}
}  // namespace Test
}  // namespace Lotus

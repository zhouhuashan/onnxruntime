// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/framework/data_types.h"
#include "gtest/gtest.h"
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace Test {

template <TensorProto_DataType T>
struct TensorTypeProto : TypeProto {
  TensorTypeProto() {
    mutable_tensor_type()->set_elem_type(T);
  }
};
template <TensorProto_DataType key, TensorProto_DataType value>
struct MapTyeProto : TypeProto {
  MapTyeProto() {
    mutable_map_type()->set_key_type(key);
    mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(value);
  }
};

TEST(DataTypeTest, MapStringStringTest) {
  MapTyeProto<TensorProto_DataType_STRING, TensorProto_DataType_STRING> maps2s_type;
  MapTyeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToString>()->IsCompatible(maps2s_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToString>()->IsCompatible(maps2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToString>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, MapStringInt64Test) {
  MapTyeProto<TensorProto_DataType_STRING, TensorProto_DataType_STRING> maps2s_type;
  MapTyeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToInt64>()->IsCompatible(maps2s_type));
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToInt64>()->IsCompatible(maps2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToInt64>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, MapStringFloatTest) {
  MapTyeProto<TensorProto_DataType_STRING, TensorProto_DataType_FLOAT> maps2f_type;
  MapTyeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToFloat>()->IsCompatible(maps2f_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToFloat>()->IsCompatible(maps2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToFloat>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, MapStringDoubleTest) {
  MapTyeProto<TensorProto_DataType_STRING, TensorProto_DataType_DOUBLE> maps2d_type;
  MapTyeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToDouble>()->IsCompatible(maps2d_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToDouble>()->IsCompatible(maps2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToDouble>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, MapInt64StringTest) {
  MapTyeProto<TensorProto_DataType_INT64, TensorProto_DataType_STRING> mapi2s_type;
  MapTyeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(mapi2s_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(mapi2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, MapInt64DoubleTest) {
  MapTyeProto<TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE> mapi2d_type;
  MapTyeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapInt64ToDouble>()->IsCompatible(mapi2d_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(mapi2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, VectorMapStringToFloatTest) {
  TypeProto type_proto;
  type_proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->set_key_type(TensorProto_DataType_STRING);
  type_proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  MapTyeProto<TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE> mapi2d_type;
  MapTyeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;

  EXPECT_TRUE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(type_proto));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(mapi2d_type));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(mapi2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, VectorMapInt64ToFloatTest) {
  TypeProto type_proto;
  type_proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->set_key_type(TensorProto_DataType_INT64);
  type_proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  MapTyeProto<TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE> mapi2d_type;
  MapTyeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;

  EXPECT_TRUE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(type_proto));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(mapi2d_type));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(mapi2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(tensor_type));
}

}  // namespace Test
}  // namespace onnxruntime

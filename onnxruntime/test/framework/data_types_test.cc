#include <typeinfo>

#include "core/framework/data_types.h"
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "onnx/defs/data_type_utils.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {

template <typename K, typename V>
struct TestMap {
  using key_type = K;
  using mapped_type = V;
};

// Try recursive type registration and compatibility tests
using TestMapToMapInt64ToFloat = TestMap<int64_t, MapInt64ToFloat>;
LOTUS_REGISTER_MAP(TestMapToMapInt64ToFloat);
using TestMapStringToVectorInt64 = TestMap<std::string, VectorInt64>;
LOTUS_REGISTER_MAP(TestMapStringToVectorInt64);

// Trial to see if we resolve the setter properly
// a map with a key that has not been registered in data_types.cc
using TestMapMLFloat16ToFloat = TestMap<MLFloat16, float>;
LOTUS_REGISTER_MAP(TestMapMLFloat16ToFloat);

template <typename T>
struct TestSequence {
  using value_type = T;
};

using TestSequenceOfSequence = TestSequence<VectorString>;
LOTUS_REGISTER_SEQ(TestSequenceOfSequence);

/// Adding an Opaque type with type parameters
struct TestOpaqueType {};
// String arrays must be extern to make them unique
// so the instantiated template would produce a unique type as well.
extern const char TestOpaqueDomain[] = "test_domain";
extern const char TestOpaqueName[] = "test_name";

using OpaqueType_1 = OpaqueRegister<TestOpaqueType, TestOpaqueDomain, TestOpaqueName>;
LOTUS_REGISTER_OPAQUE_TYPE(OpaqueType_1);
using OpaqueType_2 = OpaqueRegister<TestOpaqueType, TestOpaqueDomain, TestOpaqueName, uint64_t, float>;
LOTUS_REGISTER_OPAQUE_TYPE(OpaqueType_2);

extern const char TestOpaqueDomain_2[] = "test_doma_2";
extern const char TestOpaqueName_2[] = "test_na_2";

using OpaqueType_3 = OpaqueRegister<TestOpaqueType, TestOpaqueDomain_2, TestOpaqueName_2>;
LOTUS_REGISTER_OPAQUE_TYPE(OpaqueType_3);

// Register Maps using Opaque types as values. Note that we
// use the same cpp runtime types but due to Opaque type domain, name
// and optional parameters we produce separate MLDataTypes that are NOT
// compatible with each other.
using MyOpaqueMapCpp = std::unordered_map<int64_t, TestOpaqueType>;
using MyOpaqueMap_1 = TypeRegister<MyOpaqueMapCpp, OpaqueType_1>;
LOTUS_REGISTER_MAP(MyOpaqueMap_1);
using MyOpaqueMap_2 = TypeRegister<MyOpaqueMapCpp, OpaqueType_2>;
LOTUS_REGISTER_MAP(MyOpaqueMap_2);
using MyOpaqueMap_3 = TypeRegister<MyOpaqueMapCpp, OpaqueType_3>;
LOTUS_REGISTER_MAP(MyOpaqueMap_3);

// Register Sequence as containing an Opaque type
using MyOpaqueSeqCpp = std::vector<TestOpaqueType>;
using MyOpaqueSeq_1 = TypeRegister<MyOpaqueSeqCpp, OpaqueType_1>;
LOTUS_REGISTER_SEQ(MyOpaqueSeq_1);
using MyOpaqueSeq_2 = TypeRegister<MyOpaqueSeqCpp, OpaqueType_2>;
LOTUS_REGISTER_SEQ(MyOpaqueSeq_2);
using MyOpaqueSeq_3 = TypeRegister<MyOpaqueSeqCpp, OpaqueType_3>;
LOTUS_REGISTER_SEQ(MyOpaqueSeq_3);

// Use of Opaque types in recursive definitions. I.e. we would like to use
// it within Maps(values) and Sequences(Values) and it should work properly
// Use the example.

namespace test {

using namespace ONNX_NAMESPACE;

template <TensorProto_DataType T>
struct TensorTypeProto : TypeProto {
  TensorTypeProto() {
    mutable_tensor_type()->set_elem_type(T);
  }
};
template <TensorProto_DataType key, TensorProto_DataType value>
struct MapTypeProto : TypeProto {
  MapTypeProto() {
    mutable_map_type()->set_key_type(key);
    mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(value);
  }
};

// TODO: Add tests with Opaque type within Maps and Sequences.
TEST(DataTypeTest, OpaqueRegistrationTest) {
  // No parameters
  TypeProto opaque_proto_1;
  auto* mop = opaque_proto_1.mutable_opaque_type();
  mop->mutable_domain()->assign(TestOpaqueDomain);
  mop->mutable_name()->assign(TestOpaqueName);

  EXPECT_TRUE(DataTypeImpl::GetType<OpaqueType_1>()->IsCompatible(opaque_proto_1));
  // OpaqueType_2 has the same domain and name but also has parameters
  // so it is not compatible
  EXPECT_FALSE(DataTypeImpl::GetType<OpaqueType_2>()->IsCompatible(opaque_proto_1));
  // Even though the OpaqueType_3 has no parameters and its domain and names
  // are of the same length they are not supposed to be compatible
  EXPECT_FALSE(DataTypeImpl::GetType<OpaqueType_3>()->IsCompatible(opaque_proto_1));

  // Now change domain and name for that of OpaqueType_3
  // now we are supposed to be compatible with OpaqueType_2 but not
  // OpaqueType_1
  mop->mutable_domain()->assign(TestOpaqueDomain_2);
  mop->mutable_name()->assign(TestOpaqueName_2);
  EXPECT_FALSE(DataTypeImpl::GetType<OpaqueType_1>()->IsCompatible(opaque_proto_1));
  EXPECT_FALSE(DataTypeImpl::GetType<OpaqueType_2>()->IsCompatible(opaque_proto_1));
  EXPECT_TRUE(DataTypeImpl::GetType<OpaqueType_3>()->IsCompatible(opaque_proto_1));

  // assign back original domain/name and add params
  mop->mutable_domain()->assign(TestOpaqueDomain);
  mop->mutable_name()->assign(TestOpaqueName);
  mop->add_parameters()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_UINT64);
  mop->add_parameters()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  EXPECT_FALSE(DataTypeImpl::GetType<OpaqueType_1>()->IsCompatible(opaque_proto_1));
  EXPECT_TRUE(DataTypeImpl::GetType<OpaqueType_2>()->IsCompatible(opaque_proto_1));
  EXPECT_FALSE(DataTypeImpl::GetType<OpaqueType_3>()->IsCompatible(opaque_proto_1));
}

TEST(DataTypeTest, MapStringStringTest) {
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetTensorType<float>()->IsCompatible(tensor_type));
  EXPECT_FALSE(DataTypeImpl::GetTensorType<uint64_t>()->IsCompatible(tensor_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToString>()->IsCompatible(tensor_type));

  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_STRING> maps2s_type;
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToString>()->IsCompatible(maps2s_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToString>()->IsCompatible(maps2i_type));
}

TEST(DataTypeTest, MapStringInt64Test) {
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_STRING> maps2s_type;
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToInt64>()->IsCompatible(maps2s_type));
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToInt64>()->IsCompatible(maps2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToInt64>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, MapStringFloatTest) {
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_FLOAT> maps2f_type;
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToFloat>()->IsCompatible(maps2f_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToFloat>()->IsCompatible(maps2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToFloat>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, MapStringDoubleTest) {
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_DOUBLE> maps2d_type;
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToDouble>()->IsCompatible(maps2d_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToDouble>()->IsCompatible(maps2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToDouble>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, MapInt64StringTest) {
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_STRING> mapi2s_type;
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(mapi2s_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(mapi2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, MapInt64DoubleTest) {
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE> mapi2d_type;
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapInt64ToDouble>()->IsCompatible(mapi2d_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(mapi2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, RecursiveMapTest) {
  TypeProto map_int64_to_map_int64_to_float;
  auto* mut_map = map_int64_to_map_int64_to_float.mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_INT64);
  mut_map = mut_map->mutable_value_type()->mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_INT64);
  mut_map->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  TypeProto map_string_to_vector_of_int64;
  mut_map = map_string_to_vector_of_int64.mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_STRING);
  mut_map->mutable_value_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  EXPECT_TRUE(DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->IsCompatible(map_int64_to_map_int64_to_float));
  EXPECT_FALSE(DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->IsCompatible(map_string_to_vector_of_int64));

  EXPECT_TRUE(DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->IsCompatible(map_int64_to_map_int64_to_float));
  EXPECT_FALSE(DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->IsCompatible(map_string_to_vector_of_int64));

  // Map that contains an Opaque_1
  const auto* op1_proto = DataTypeImpl::GetType<OpaqueType_1>();
  TypeProto unod_map_int64_to_op1;
  mut_map = unod_map_int64_to_op1.mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_INT64);
  mut_map->mutable_value_type()->CopyFrom(*op1_proto->GetTypeProto());
  EXPECT_TRUE(DataTypeImpl::GetType<MyOpaqueMap_1>()->IsCompatible(unod_map_int64_to_op1));

  // Map that contains an Opaque_2
  const auto* op2_proto = DataTypeImpl::GetType<OpaqueType_2>();
  TypeProto unod_map_int64_to_op2;
  mut_map = unod_map_int64_to_op2.mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_INT64);
  mut_map->mutable_value_type()->CopyFrom(*op2_proto->GetTypeProto());
  EXPECT_TRUE(DataTypeImpl::GetType<MyOpaqueMap_2>()->IsCompatible(unod_map_int64_to_op2));

  // Map that contains an Opaque_3
  const auto* op3_proto = DataTypeImpl::GetType<OpaqueType_3>();
  TypeProto unod_map_int64_to_op3;
  mut_map = unod_map_int64_to_op3.mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_INT64);
  mut_map->mutable_value_type()->CopyFrom(*op3_proto->GetTypeProto());
  EXPECT_TRUE(DataTypeImpl::GetType<MyOpaqueMap_3>()->IsCompatible(unod_map_int64_to_op3));
}

TEST(DataTypeTest, RecursiveVectorTest) {
  TypeProto seq_of_seq_string;
  auto* mut_seq = seq_of_seq_string.mutable_sequence_type();
  mut_seq = mut_seq->mutable_elem_type()->mutable_sequence_type();
  mut_seq->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);

  EXPECT_TRUE(DataTypeImpl::GetType<TestSequenceOfSequence>()->IsCompatible(seq_of_seq_string));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(seq_of_seq_string));
}

TEST(DataTypeTest, VectorMapStringToFloatTest) {
  TypeProto vector_map_string_to_float;
  vector_map_string_to_float.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->set_key_type(TensorProto_DataType_STRING);
  vector_map_string_to_float.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE> mapi2d_type;
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;

  EXPECT_TRUE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(vector_map_string_to_float));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(mapi2d_type));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(mapi2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, VectorMapInt64ToFloatTest) {
  TypeProto type_proto;
  type_proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->set_key_type(TensorProto_DataType_INT64);
  type_proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE> mapi2d_type;
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;

  EXPECT_TRUE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(type_proto));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(mapi2d_type));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(mapi2i_type));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(tensor_type));
}

TEST(DataTypeTest, DataUtilsTest) {
  using namespace ONNX_NAMESPACE::Utils;
  // Test Tensor
  {
    const auto* ten_proto = DataTypeImpl::GetTensorType<uint64_t>()->GetTypeProto();
    EXPECT_NE(ten_proto, nullptr);
    DataType ten_dt = DataTypeUtils::ToType(*ten_proto);
    EXPECT_NE(ten_dt, nullptr);
    DataType ten_from_str = DataTypeUtils::ToType(*ten_dt);
    // Expect internalized strings
    EXPECT_EQ(ten_dt, ten_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(ten_dt);
    EXPECT_TRUE(DataTypeImpl::GetTensorType<uint64_t>()->IsCompatible(from_dt_proto));
  }
  // Test Simple map
  {
    const auto* map_proto = DataTypeImpl::GetType<MapStringToString>()->GetTypeProto();
    EXPECT_NE(map_proto, nullptr);
    DataType map_dt = DataTypeUtils::ToType(*map_proto);
    EXPECT_NE(map_dt, nullptr);
    DataType map_from_str = DataTypeUtils::ToType(*map_dt);
    // Expect internalized strings
    EXPECT_EQ(map_dt, map_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(map_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<MapStringToString>()->IsCompatible(from_dt_proto));
  }
  // Test map with recursive value
  {
    const auto* map_proto = DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->GetTypeProto();
    EXPECT_NE(map_proto, nullptr);
    DataType map_dt = DataTypeUtils::ToType(*map_proto);
    EXPECT_NE(map_dt, nullptr);
    DataType map_from_str = DataTypeUtils::ToType(*map_dt);
    // Expect internalized strings
    EXPECT_EQ(map_dt, map_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(map_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->IsCompatible(from_dt_proto));
  }
  // Test simple seq
  {
    const auto* seq_proto = DataTypeImpl::GetType<VectorFloat>()->GetTypeProto();
    EXPECT_NE(seq_proto, nullptr);
    DataType seq_dt = DataTypeUtils::ToType(*seq_proto);
    EXPECT_NE(seq_dt, nullptr);
    DataType seq_from_str = DataTypeUtils::ToType(*seq_dt);
    // Expect internalized strings
    EXPECT_EQ(seq_dt, seq_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(seq_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<VectorFloat>()->IsCompatible(from_dt_proto));
  }
  // Test Sequence with recursion
  {
    const auto* seq_proto = DataTypeImpl::GetType<VectorMapStringToFloat>()->GetTypeProto();
    EXPECT_NE(seq_proto, nullptr);
    DataType seq_dt = DataTypeUtils::ToType(*seq_proto);
    EXPECT_NE(seq_dt, nullptr);
    DataType seq_from_str = DataTypeUtils::ToType(*seq_dt);
    // Expect internalized strings
    EXPECT_EQ(seq_dt, seq_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(seq_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(from_dt_proto));
  }
  // Test Opaque type no parameters
  {
    const auto* op_proto = DataTypeImpl::GetType<OpaqueType_1>()->GetTypeProto();
    EXPECT_NE(op_proto, nullptr);
    DataType op_dt = DataTypeUtils::ToType(*op_proto);
    EXPECT_NE(op_dt, nullptr);
    DataType op_from_str = DataTypeUtils::ToType(*op_dt);
    // Expect internalized strings
    EXPECT_EQ(op_dt, op_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(op_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<OpaqueType_1>()->IsCompatible(from_dt_proto));
  }
  // Test Opaque type with parameters
  {
    const auto* op_proto = DataTypeImpl::GetType<OpaqueType_2>()->GetTypeProto();
    EXPECT_NE(op_proto, nullptr);
    DataType op_dt = DataTypeUtils::ToType(*op_proto);
    EXPECT_NE(op_dt, nullptr);
    DataType op_from_str = DataTypeUtils::ToType(*op_dt);
    // Expect internalized strings
    EXPECT_EQ(op_dt, op_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(op_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<OpaqueType_2>()->IsCompatible(from_dt_proto));
  }
}

}  // namespace test
}  // namespace onnxruntime

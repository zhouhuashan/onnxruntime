#ifdef _MSC_VER
#pragma warning(push)
// 'identifier' : unreferenced formal parameter
#pragma warning(disable: 4100)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable: 4800)
#endif
#include "google/protobuf/util/message_differencer.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "core/graph/utils.h"
#include "core/protobuf/onnx-ml.pb.h"
#include "gtest/gtest.h"

using google::protobuf::util::MessageDifferencer;
using LotusIR::Utils::OpUtils;
using namespace onnx;

namespace LotusIR
{
    namespace Test
    {
        TEST(OpUtilsTest, SplitRecords)
        {
            std::vector<Utils::StringRange> tokens;
            Utils::StringRange s("r1:tensor(int32),r2:seq(tensor(double)),r3:record(a1:tensor(string),a2:tensor(int32)),r4:sparse(int32)");
            OpUtils::SplitStringTokens(s, tokens);
            std::string s1 = std::string(tokens[0].Data(), tokens[0].Size());
            EXPECT_EQ(s1, "r1:tensor(int32)");
            std::string s2 = std::string(tokens[1].Data(), tokens[1].Size());
            EXPECT_EQ(s2, "r2:seq(tensor(double))");
            std::string s3 = std::string(tokens[2].Data(), tokens[2].Size());
            EXPECT_EQ(s3, "r3:record(a1:tensor(string),a2:tensor(int32))");
            std::string s4 = std::string(tokens[3].Data(), tokens[3].Size());
            EXPECT_EQ(s4, "r4:sparse(int32)");
        }

        TEST(OpUtilsTest, TestPTYPE)
        {
            PTYPE p1 = OpUtils::ToType("tensor(int32)");
            PTYPE p2 = OpUtils::ToType("tensor(int32)");
            PTYPE p3 = OpUtils::ToType("tensor(int32)");
            EXPECT_EQ(p1, p2);
            EXPECT_EQ(p2, p3);
            EXPECT_EQ(p1, p3);
            PTYPE p4 = OpUtils::ToType("seq(tensor(int32))");
            PTYPE p5 = OpUtils::ToType("seq(tensor(int32))");
            PTYPE p6 = OpUtils::ToType("seq(tensor(int32))");
            EXPECT_EQ(p4, p5);
            EXPECT_EQ(p5, p6);
            EXPECT_EQ(p4, p6);

            TypeProto t1 = OpUtils::ToTypeProto(p1);
            EXPECT_TRUE(t1.has_tensor_type());
            EXPECT_TRUE(t1.tensor_type().has_elem_type());
            EXPECT_EQ(t1.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
            TypeProto t2 = OpUtils::ToTypeProto(p2);
            EXPECT_TRUE(t2.has_tensor_type());
            EXPECT_TRUE(t2.tensor_type().has_elem_type());
            EXPECT_EQ(t2.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
            TypeProto t3 = OpUtils::ToTypeProto(p3);
            EXPECT_TRUE(t3.has_tensor_type());
            EXPECT_TRUE(t3.tensor_type().has_elem_type());
            EXPECT_EQ(t3.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
            TypeProto t4 = Utils::OpUtils::ToTypeProto(p4);
            EXPECT_TRUE(t4.has_sequence_type());
            EXPECT_TRUE(t4.sequence_type().has_elem_type());
            EXPECT_TRUE(t4.sequence_type().elem_type().has_tensor_type());
            EXPECT_TRUE(t4.sequence_type().elem_type().tensor_type().has_elem_type());
            EXPECT_EQ(t4.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
            TypeProto t5 = Utils::OpUtils::ToTypeProto(p5);
            EXPECT_TRUE(t5.has_sequence_type());
            EXPECT_TRUE(t5.sequence_type().has_elem_type());
            EXPECT_TRUE(t5.sequence_type().elem_type().has_tensor_type());
            EXPECT_TRUE(t5.sequence_type().elem_type().tensor_type().has_elem_type());
            EXPECT_EQ(t5.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
            TypeProto t6 = Utils::OpUtils::ToTypeProto(p6);
            EXPECT_TRUE(t6.has_sequence_type());
            EXPECT_TRUE(t6.sequence_type().has_elem_type());
            EXPECT_TRUE(t6.sequence_type().elem_type().has_tensor_type());
            EXPECT_TRUE(t6.sequence_type().elem_type().tensor_type().has_elem_type());
            EXPECT_EQ(t6.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
        }

        TEST(OpUtilsTest, ToStringTest)
        {
            TypeProto scalar;
            TypeProto_Tensor* tensor = scalar.mutable_tensor_type();
            tensor->set_elem_type(TensorProto_DataType_INT32);
            tensor->mutable_shape();
            EXPECT_EQ(OpUtils::ToString(scalar), "int32");

            TypeProto t;
            t.mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
            EXPECT_EQ(OpUtils::ToString(t), "tensor(float)");

            TypeProto seq;
            seq.mutable_sequence_type()->mutable_elem_type()->mutable_sequence_type()->mutable_elem_type()->
                mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
            EXPECT_EQ(OpUtils::ToString(seq), "seq(seq(tensor(float)))");

            TypeProto map1;
            map1.mutable_map_type()->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            map1.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);
            EXPECT_EQ(OpUtils::ToString(map1), "map(string,tensor(string))");

            TypeProto map2;
            map2.mutable_map_type()->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            map2.mutable_map_type()->mutable_value_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);
            EXPECT_EQ(OpUtils::ToString(map2), "map(string,seq(tensor(string)))");
        }

        TEST(OpUtilsTest, FromStringTest)
        {
            TypeProto scalar1;
            OpUtils::FromString("int32", scalar1);
            TypeProto scalar2;
            TypeProto_Tensor* tensor = scalar2.mutable_tensor_type();
            tensor->set_elem_type(TensorProto_DataType_INT32);
            tensor->mutable_shape();
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(scalar1, scalar2));

            TypeProto t1;
            OpUtils::FromString("tensor(float)", t1);
            TypeProto t2;
            t2.mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(t1, t2));

            TypeProto seq1;
            OpUtils::FromString("seq(tensor(float))", seq1);
            TypeProto seq2;
            seq2.mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(seq1, seq2));

            TypeProto map1;
            OpUtils::FromString("map(string,tensor(int32))", map1);
            TypeProto map2;
            map2.mutable_map_type()->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            map2.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(map1, map2));

            TypeProto map3;
            OpUtils::FromString("map(string,seq(tensor(int32)))", map3);
            TypeProto map4;
            map4.mutable_map_type()->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            map4.mutable_map_type()->mutable_value_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(map3, map4));
        }
    }
}

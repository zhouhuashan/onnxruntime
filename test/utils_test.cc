#include "google/protobuf/util/message_differencer.h"
#include "core/protobuf/graph.pb.h"
#include "gtest/gtest.h"
#include "utils.h"

using google::protobuf::util::MessageDifferencer;
using LotusIR::Utils::OpUtils;

namespace LotusIR
{
    namespace Test
    {
        TEST(OpUtilsTest, TestPTYPE)
        {
            PTYPE p1 = OpUtils::ToType("int32");
            PTYPE p2 = OpUtils::ToType("int32");
            PTYPE p3 = OpUtils::ToType("int32");
            EXPECT_EQ(p1, p2);
            EXPECT_EQ(p2, p3);
            EXPECT_EQ(p1, p3);
            PTYPE p4 = OpUtils::ToType("seq(int32)");
            PTYPE p5 = OpUtils::ToType("seq(int32)");
            PTYPE p6 = OpUtils::ToType("seq(int32)");
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
            EXPECT_TRUE(t4.has_seq_type());
            EXPECT_TRUE(t4.seq_type().has_elem_type());
            EXPECT_TRUE(t4.seq_type().elem_type().has_tensor_type());
            EXPECT_TRUE(t4.seq_type().elem_type().tensor_type().has_elem_type());
            EXPECT_EQ(t4.seq_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
            TypeProto t5 = Utils::OpUtils::ToTypeProto(p5);
            EXPECT_TRUE(t5.has_seq_type());
            EXPECT_TRUE(t5.seq_type().has_elem_type());
            EXPECT_TRUE(t5.seq_type().elem_type().has_tensor_type());
            EXPECT_TRUE(t5.seq_type().elem_type().tensor_type().has_elem_type());
            EXPECT_EQ(t5.seq_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
            TypeProto t6 = Utils::OpUtils::ToTypeProto(p6);
            EXPECT_TRUE(t6.has_seq_type());
            EXPECT_TRUE(t6.seq_type().has_elem_type());
            EXPECT_TRUE(t6.seq_type().elem_type().has_tensor_type());
            EXPECT_TRUE(t6.seq_type().elem_type().tensor_type().has_elem_type());
            EXPECT_EQ(t6.seq_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
        }

        TEST(OpUtilsTest, ToStringTest)
        {
            TypeProto h;
            h.mutable_handle_type();
            EXPECT_EQ(OpUtils::ToString(h), "handle");

            TypeProto s;
            s.mutable_sparse_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
            EXPECT_EQ(OpUtils::ToString(s), "sparse(float)");

            TypeProto t;
            t.mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
            EXPECT_EQ(OpUtils::ToString(t), "float");

            TypeProto seq;
            seq.mutable_seq_type()->mutable_elem_type()->mutable_seq_type()->mutable_elem_type()->
                mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
            EXPECT_EQ(OpUtils::ToString(seq), "seq(seq(float))");

            TypeProto tuple;
            tuple.mutable_tuple_type()->mutable_elem_type()->Add()->mutable_handle_type();
            tuple.mutable_tuple_type()->mutable_elem_type()->Add()->mutable_handle_type();
            tuple.mutable_tuple_type()->mutable_elem_type()->Add()->mutable_handle_type();
            EXPECT_EQ(OpUtils::ToString(tuple), "tuple(handle,handle,handle)");

            TypeProto map;
            map.mutable_map_type()->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            map.mutable_map_type()->set_value_type(TensorProto_DataType::TensorProto_DataType_STRING);
            EXPECT_EQ(OpUtils::ToString(map), "map(string,string)");
        }

        TEST(OpUtilsTest, FromStringTest)
        {
            TypeProto h1;
            OpUtils::FromString("handle", h1);
            TypeProto h2;
            h2.mutable_handle_type();
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(h1, h2));

            TypeProto s1;
            OpUtils::FromString("sparse(int32)", s1);
            TypeProto s2;
            s2.mutable_sparse_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_INT32);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(s1, s2));

            TypeProto t1;
            OpUtils::FromString("float", t1);
            TypeProto t2;
            t2.mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(t1, t2));

            TypeProto seq1;
            OpUtils::FromString("seq(float)", seq1);
            TypeProto seq2;
            seq2.mutable_seq_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(seq1, seq2));

            TypeProto tuple1;
            OpUtils::FromString("tuple(int32,sparse(int32),handle)", tuple1);
            TypeProto tuple2;
            tuple2.mutable_tuple_type()->mutable_elem_type()->Add()->mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_INT32);
            tuple2.mutable_tuple_type()->mutable_elem_type()->Add()->mutable_sparse_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            tuple2.mutable_tuple_type()->mutable_elem_type()->Add()->mutable_handle_type();
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(tuple1, tuple2));

            TypeProto tuple3;
            OpUtils::FromString("tuple(seq(int32))", tuple3);
            TypeProto tuple4;
            tuple4.mutable_tuple_type()->mutable_elem_type()->Add()->mutable_seq_type()->
                mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_INT32);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(tuple3, tuple4));

            TypeProto map1;
            OpUtils::FromString("map(string,int32)", map1);
            TypeProto map2;
            map2.mutable_map_type()->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            map2.mutable_map_type()->set_value_type(TensorProto_DataType::TensorProto_DataType_INT32);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(map1, map2));
        }
    }
}

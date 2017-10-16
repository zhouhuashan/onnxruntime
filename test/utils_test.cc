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
#include "core/protobuf/graph.pb.h"
#include "gtest/gtest.h"

using google::protobuf::util::MessageDifferencer;
using LotusIR::Utils::OpUtils;

namespace LotusIR
{
    namespace Test
    {
        TEST(OpUtilsTest, SplitRecords)
        {
            std::vector<Utils::StringRange> tokens;
            Utils::StringRange s("r1:int32,r2:seq(double),r3:record(a1:string,a2:int32),r4:sparse(int32)");
            OpUtils::SplitRecords(s, tokens);
            std::string s1 = std::string(tokens[0].Data(), tokens[0].Size());
            EXPECT_EQ(s1, "r1:int32");
            std::string s2 = std::string(tokens[1].Data(), tokens[1].Size());
            EXPECT_EQ(s2, "r2:seq(double)");
            std::string s3 = std::string(tokens[2].Data(), tokens[2].Size());
            EXPECT_EQ(s3, "r3:record(a1:string,a2:int32)");
            std::string s4 = std::string(tokens[3].Data(), tokens[3].Size());
            EXPECT_EQ(s4, "r4:sparse(int32)");
        }

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

            TypeProto map1;
            map1.mutable_map_type()->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            map1.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);
            EXPECT_EQ(OpUtils::ToString(map1), "map(string,string)");

            TypeProto map2;
            map2.mutable_map_type()->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            map2.mutable_map_type()->mutable_value_type()->mutable_seq_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);
            EXPECT_EQ(OpUtils::ToString(map2), "map(string,seq(string))");

            TypeProto record1;
            ValueInfoProto* v1 = record1.mutable_record_type()->mutable_field()->Add();
            v1->set_name("r1");
            v1->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            ValueInfoProto* v2 = record1.mutable_record_type()->mutable_field()->Add();
            v2->set_name("r2");
            v2->mutable_type()->mutable_seq_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_DOUBLE);
            ValueInfoProto* v3 = record1.mutable_record_type()->mutable_field()->Add();
            v3->set_name("r3");
            TypeProto_MapTypeProto* m = v3->mutable_type()->mutable_map_type();
            m->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            m->mutable_value_type()->mutable_seq_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);
            EXPECT_EQ(OpUtils::ToString(record1), "record(r1:int32,r2:seq(double),r3:map(string,seq(string)))");

            TypeProto union1;
            ValueInfoProto* uv1 = union1.mutable_union_type()->mutable_choice()->Add();
            uv1->set_name("c1");
            uv1->mutable_type()->mutable_seq_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);
            ValueInfoProto* uv2 = union1.mutable_union_type()->mutable_choice()->Add();
            uv2->set_name("c2");
            uv2->mutable_type()->mutable_sparse_tensor_type()->set_elem_type(TensorProto_DataType_DOUBLE);
            ValueInfoProto* uv3 = union1.mutable_union_type()->mutable_choice()->Add();
            uv3->set_name("c3");
            TypeProto_MapTypeProto* m1 = uv3->mutable_type()->mutable_map_type();
            m1->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            m1->mutable_value_type()->mutable_seq_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);
            EXPECT_EQ(OpUtils::ToString(union1), "union(c1:seq(string),c2:sparse(double),c3:map(string,seq(string)))");
        }

        TEST(OpUtilsTest, FromStringTest)
        {
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

            TypeProto map1;
            OpUtils::FromString("map(string,int32)", map1);
            TypeProto map2;
            map2.mutable_map_type()->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            map2.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(map1, map2));

            TypeProto map3;
            OpUtils::FromString("map(string,seq(int32))", map3);
            TypeProto map4;
            map4.mutable_map_type()->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            map4.mutable_map_type()->mutable_value_type()->mutable_seq_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(map3, map4));

            TypeProto record1;
            // Test parsing random whitespaces in between.
            OpUtils::FromString("record (r1: int32,r2:seq( double), r3:map(string , record(a1:string,a2: int32) ), r4:record(x1:int32))", record1);
            TypeProto record2;
            ValueInfoProto* v1 = record2.mutable_record_type()->mutable_field()->Add();
            v1->set_name("r1");
            v1->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            ValueInfoProto* v2 = record2.mutable_record_type()->mutable_field()->Add();
            v2->set_name("r2");
            v2->mutable_type()->mutable_seq_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_DOUBLE);
            ValueInfoProto* v3 = record2.mutable_record_type()->mutable_field()->Add();
            v3->set_name("r3");
            TypeProto_MapTypeProto* m = v3->mutable_type()->mutable_map_type();
            m->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            ValueInfoProto* a1 = m->mutable_value_type()->mutable_record_type()->mutable_field()->Add();
            a1->set_name("a1");
            a1->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);
            ValueInfoProto* a2 = m->mutable_value_type()->mutable_record_type()->mutable_field()->Add();
            a2->set_name("a2");
            a2->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            ValueInfoProto* v4 = record2.mutable_record_type()->mutable_field()->Add();
            v4->set_name("r4");
            ValueInfoProto* x1 = v4->mutable_type()->mutable_record_type()->mutable_field()->Add();
            x1->set_name("x1");
            x1->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(record1, record2));

            TypeProto union1;
            // Test parsing random whitespaces in between.
            OpUtils::FromString("union (c1 : seq ( string ), c2: sparse( double ),c3 : map ( string ,seq( string) ))", union1);
            TypeProto union2;
            ValueInfoProto* uv1 = union2.mutable_union_type()->mutable_choice()->Add();
            uv1->set_name("c1");
            uv1->mutable_type()->mutable_seq_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);
            ValueInfoProto* uv2 = union2.mutable_union_type()->mutable_choice()->Add();
            uv2->set_name("c2");
            uv2->mutable_type()->mutable_sparse_tensor_type()->set_elem_type(TensorProto_DataType_DOUBLE);
            ValueInfoProto* uv3 = union2.mutable_union_type()->mutable_choice()->Add();
            uv3->set_name("c3");
            TypeProto_MapTypeProto* m1 = uv3->mutable_type()->mutable_map_type();
            m1->set_key_type(TensorProto_DataType::TensorProto_DataType_STRING);
            m1->mutable_value_type()->mutable_seq_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);
            EXPECT_TRUE(MessageDifferencer::MessageDifferencer::Equals(union1, union2));
        }
    }
}

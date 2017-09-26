#include <iostream>
#include "core/protobuf/Type.pb.h"
#include "gtest/gtest.h"
#include "op.h"
#include "utils.h"

namespace LotusIR
{
    namespace Test
    {
        TEST(FormalParamTest, Success)
        {
            OperatorSchema::FormalParameter p("input", "int32", "desc: integer input");
            EXPECT_EQ("input", p.GetName());
            EXPECT_EQ("int32", p.GetTypeStr());
            EXPECT_EQ("desc: integer input", p.GetDescription());
            EXPECT_EQ(Utils::OpUtils::ToType("int32"), *p.GetTypes().begin());
        }

        TEST(OpRegistrationTest, OpRegTestNoTypes)
        {
            REGISTER_OP(__TestOpRegNoTypes).Description("Op Registration Without Types.")
                .Input("input_1", "docstr for input_1.")
                .Input("input_2", "docstr for input_2.")
                .Output("output_1", "docstr for output_1.");
            const OperatorSchema* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestOpRegNoTypes", &op);
            EXPECT_TRUE(success);
            EXPECT_EQ(op->GetInputs().size(), 2);
            EXPECT_EQ(op->GetInputs()[0].GetName(), "input_1");
            EXPECT_EQ(op->GetInputs()[0].GetTypes().size(), 0);
            EXPECT_EQ(op->GetInputs()[1].GetName(), "input_2");
            EXPECT_EQ(op->GetInputs()[1].GetTypes().size(), 0);
            EXPECT_EQ(op->GetOutputs().size(), 1);
            EXPECT_EQ(op->GetOutputs()[0].GetName(), "output_1");
            EXPECT_EQ(op->GetOutputs()[0].GetTypes().size(), 0);
        }

        TEST(OpRegistrationTest, OpRegTest)
        {
            REGISTER_OP(__TestOpReg).Description("Op Registration Basic Test.")
                .Input("input_1", "docstr for input_1.", "int32")
                .Input("input_2", "docstr for input_2.", "int32")
                .Output("output_1", "docstr for output_1.", "int32");
            const OperatorSchema* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestOpReg", &op);
            EXPECT_TRUE(success);

            EXPECT_EQ(op->GetInputs().size(), 2);
            EXPECT_EQ(op->GetInputs()[0].GetName(), "input_1");
            EXPECT_EQ(op->GetInputs()[0].GetTypes().size(), 1);
            EXPECT_EQ(**op->GetInputs()[0].GetTypes().find(Utils::OpUtils::ToType("int32")), "int32");
            EXPECT_EQ(op->GetInputs()[1].GetName(), "input_2");
            EXPECT_EQ(op->GetInputs()[1].GetTypes().size(), 1);
            EXPECT_EQ(**op->GetInputs()[1].GetTypes().find(Utils::OpUtils::ToType("int32")), "int32");
            EXPECT_EQ(op->GetOutputs().size(), 1);
            EXPECT_EQ(op->GetOutputs()[0].GetName(), "output_1");
            EXPECT_EQ(op->GetOutputs()[0].GetTypes().size(), 1);
            EXPECT_EQ(**op->GetOutputs()[0].GetTypes().find(Utils::OpUtils::ToType("int32")), "int32");
        }

        TEST(OpRegistrationTest, TypeConstraintTest)
        {
            REGISTER_OP(__TestTypeConstraint).Description("Op with Type Constraint.")
                .Input("input_1", "docstr for input_1.", "T")
                .Input("input_2", "docstr for input_2.", "T")
                .Output("output_1", "docstr for output_1.", "T")
                .TypeConstraint("T", { "float16", "float", "double" }, "Constrain input and output types to floats.");
            const OperatorSchema* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestTypeConstraint", &op);
            EXPECT_TRUE(success);

            EXPECT_EQ(op->GetInputs().size(), 2);
            EXPECT_EQ(op->GetInputs()[0].GetName(), "input_1");
            EXPECT_EQ(op->GetInputs()[0].GetTypes().size(), 3);
            EXPECT_EQ(**op->GetInputs()[0].GetTypes().find(Utils::OpUtils::ToType("float16")), "float16");
            EXPECT_EQ(**op->GetInputs()[0].GetTypes().find(Utils::OpUtils::ToType("float")), "float");
            EXPECT_EQ(**op->GetInputs()[0].GetTypes().find(Utils::OpUtils::ToType("double")), "double");

            EXPECT_EQ(op->GetInputs()[1].GetName(), "input_2");
            EXPECT_EQ(op->GetInputs()[1].GetTypes().size(), 3);
            EXPECT_EQ(**op->GetInputs()[1].GetTypes().find(Utils::OpUtils::ToType("float16")), "float16");
            EXPECT_EQ(**op->GetInputs()[1].GetTypes().find(Utils::OpUtils::ToType("float")), "float");
            EXPECT_EQ(**op->GetInputs()[1].GetTypes().find(Utils::OpUtils::ToType("double")), "double");

            EXPECT_EQ(op->GetOutputs().size(), 1);
            EXPECT_EQ(op->GetOutputs()[0].GetName(), "output_1");
            EXPECT_EQ(op->GetOutputs()[0].GetTypes().size(), 3);
            EXPECT_EQ(**op->GetOutputs()[0].GetTypes().find(Utils::OpUtils::ToType("float16")), "float16");
            EXPECT_EQ(**op->GetOutputs()[0].GetTypes().find(Utils::OpUtils::ToType("float")), "float");
            EXPECT_EQ(**op->GetOutputs()[0].GetTypes().find(Utils::OpUtils::ToType("double")), "double");
        }

        TEST(OpRegistrationTest, AttributeTest)
        {
            REGISTER_OP(__TestAttr).Description("Op with attributes.")
                .Attr("my_attr_int", "attr with INT type", AttrType::INT)
                .Attr("my_attr_float", "attr with FLOAT type", AttrType::FLOAT)
                .Attr("my_attr_string", "attr with STRING type", AttrType::STRING);
            const OperatorSchema* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestAttr", &op);
            EXPECT_TRUE(success);


            std::vector<std::string> expected_strings = { "my_attr_int", "my_attr_float", "my_attr_string" };
            std::vector<AttrType> expected_types = { AttrType::INT, AttrType::FLOAT, AttrType::STRING };
            
            int size = op->GetAttributes().size();
            EXPECT_EQ(size, 3);
            for (int i = 0; i < size; i++)
            {
                EXPECT_EQ(op->GetAttributes()[i].GetName(), expected_strings[i]);
                EXPECT_EQ(op->GetAttributes()[i].GetType(), expected_types[i]);
            }
        }

        TEST(OpRegistrationTest, AttributeDefaultValueTest)
        {
            REGISTER_OP(__TestAttrDefaultValue).Description("Op with attributes that have default values")
                .Attr("my_attr_int", "attr with default value of 99.", AttrType::INT, 99LL)
                .Attr("my_attr_float", "attr with default value of 0.99.", AttrType::FLOAT, 0.99f)
                .Attr("my_attr_string", "attr with default value of \"99\".", AttrType::STRING, "99");

            const OperatorSchema* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestAttrDefaultValue", &op);
            EXPECT_TRUE(success);

            EXPECT_EQ(op->GetAttributes().size(), 3);

            EXPECT_EQ(op->GetAttributes()[0].GetName(), "my_attr_int");
            EXPECT_EQ(op->GetAttributes()[0].GetType(), AttrType::INT);
            const AttributeProto* a1;
            EXPECT_TRUE(op->GetAttributes()[0].HasDefaultValue(&a1));
            EXPECT_EQ(a1->name(), "my_attr_int");
            EXPECT_TRUE(a1->has_i());
            EXPECT_EQ(a1->i(), 99LL);

            EXPECT_EQ(op->GetAttributes()[1].GetName(), "my_attr_float");
            EXPECT_EQ(op->GetAttributes()[1].GetType(), AttrType::FLOAT);
            const AttributeProto* a2;
            EXPECT_TRUE(op->GetAttributes()[1].HasDefaultValue(&a2));
            EXPECT_EQ(a2->name(), "my_attr_float");
            EXPECT_TRUE(a2->has_f());
            EXPECT_EQ(a2->f(), 0.99f);

            EXPECT_EQ(op->GetAttributes()[2].GetName(), "my_attr_string");
            EXPECT_EQ(op->GetAttributes()[2].GetType(), AttrType::STRING);
            const AttributeProto* a3;
            EXPECT_TRUE(op->GetAttributes()[2].HasDefaultValue(&a3));
            EXPECT_EQ(a3->name(), "my_attr_string");
            EXPECT_TRUE(a3->has_s());
            EXPECT_EQ(a3->s(), "99");
        }

        TEST(OpRegistrationTest, AttributeDefaultValueListTest)
        {
            REGISTER_OP(__TestAttrDefaultValueList).Description("Op with attributes that have default list of values.")
                .Attr("my_attr_ints", "attr with default value of [98, 99, 100].", AttrType::INTS, std::vector<int64_t> {98LL, 99LL, 100LL})
                .Attr("my_attr_floats", "attr with default value of [0.98, 0.99, 1.00].", AttrType::FLOATS, std::vector<float> {0.98f, 0.99f, 1.00f})
                .Attr("my_attr_strings", "attr with default value of [\"98\", \"99\", \"100\"].", AttrType::STRINGS, std::vector<std::string> {"98", "99", "100"});

            const OperatorSchema* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestAttrDefaultValueList", &op);
            EXPECT_TRUE(success);

            EXPECT_EQ(op->GetAttributes().size(), 3);

            EXPECT_EQ(op->GetAttributes()[0].GetName(), "my_attr_ints");
            EXPECT_EQ(op->GetAttributes()[0].GetType(), AttrType::INTS);
            const AttributeProto* a1;
            EXPECT_TRUE(op->GetAttributes()[0].HasDefaultValue(&a1));
            EXPECT_EQ(a1->name(), "my_attr_ints");
            int size = a1->ints_size();
            EXPECT_EQ(size, 3);
            std::vector<int64_t> expected_ints = { 98LL, 99LL, 100LL };
            for (int i = 0; i < size; i++)
            {
                EXPECT_EQ(a1->ints(i), expected_ints[i]);
            }

            EXPECT_EQ(op->GetAttributes()[1].GetName(), "my_attr_floats");
            EXPECT_EQ(op->GetAttributes()[1].GetType(), AttrType::FLOATS);
            const AttributeProto* a2;
            EXPECT_TRUE(op->GetAttributes()[1].HasDefaultValue(&a2));
            EXPECT_EQ(a2->name(), "my_attr_floats");
            size = a2->floats_size();
            EXPECT_EQ(size, 3);
            std::vector<float> expected_floats = { 0.98f, 0.99f, 1.00f };
            for (int i = 0; i < size; i++)
            {
                EXPECT_EQ(a2->floats(i), expected_floats[i]);
            }

            EXPECT_EQ(op->GetAttributes()[2].GetName(), "my_attr_strings");
            EXPECT_EQ(op->GetAttributes()[2].GetType(), AttrType::STRINGS);
            const AttributeProto* a3;
            EXPECT_TRUE(op->GetAttributes()[2].HasDefaultValue(&a3));
            EXPECT_EQ(a3->name(), "my_attr_strings");
            size = a3->strings_size();
            EXPECT_EQ(size, 3);
            std::vector<std::string> expected_strings = { "98", "99", "100" };
            for (int i = 0; i < size; i++)
            {
                EXPECT_EQ(a3->strings(i), expected_strings[i]);
            }
        }
    }
}
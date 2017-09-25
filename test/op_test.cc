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

        TEST(OpRegistrationTest, BasicOpRegTest)
        {
            REGISTER_OP("__TestOpReg").Description("Op Registration Basic Test.")
                .Input("input_1", "int32", "docstr for input_1.")
                .Input("input_2", "int32", "docstr for input_2.")
                .Output("output_1", "int32", "docstr for output_1.");
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
            REGISTER_OP("__TestTypeConstraint").Description("Op with Type Constraint.")
                .Input("input_1", "T", "docstr for input_1.")
                .Input("input_2", "T", "docstr for input_2.")
                .Output("output_1", "T", "docstr for output_1.")
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
            REGISTER_OP("__TestAttr").Description("Op with attributes.")
                .Attr("my_attr_int", AttrType::INT, "attr with INT type")
                .Attr("my_attr_float", AttrType::FLOAT, "attr with FLOAT type")
                .Attr("my_attr_string", AttrType::STRING, "attr with STRING type");
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
            REGISTER_OP("__TestAttrDefaultValue").Description("Op with attributes that have default values")
                .Attr("my_attr_int", AttrType::INT, "attr with default value of 99.", 99LL)
                .Attr("my_attr_float", AttrType::FLOAT, "attr with default value of 0.99.", 0.99f)
                .Attr("my_attr_string", AttrType::STRING, "attr with default value of \"99\".", "99");

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
            REGISTER_OP("__TestAttrDefaultValueList").Description("Op with attributes that have default list of values.")
                .Attr("my_attr_ints", AttrType::INTS, "attr with default value of [98, 99, 100].", std::vector<int64_t> {98LL, 99LL, 100LL})
                .Attr("my_attr_floats", AttrType::FLOATS, "attr with default value of [0.98, 0.99, 1.00].", std::vector<float> {0.98f, 0.99f, 1.00f})
                .Attr("my_attr_strings", AttrType::STRINGS, "attr with default value of [\"98\", \"99\", \"100\"].", std::vector<std::string> {"98", "99", "100"});

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
#include <iostream>
#include "gtest/gtest.h"
#include "core/graph/op.h"
#include "core/graph/utils.h"
#include "external/onnx/onnx/onnx-ml.pb.h"

using namespace onnx;

namespace LotusIR
{
    namespace Test
    {
        TEST(FormalParamTest, Success)
        {
            OpSignature::FormalParameter p("input", "tensor(int32)", "desc: integer input");
            EXPECT_EQ("input", p.GetName());
            EXPECT_EQ("tensor(int32)", p.GetTypeStr());
            EXPECT_EQ("desc: integer input", p.GetDescription());
            EXPECT_EQ(Utils::OpUtils::ToType("tensor(int32)"), *p.GetTypes().begin());
        }

        TEST(OpRegistrationTest, OpRegTestNoTypes)
        {
            REGISTER_OPERATOR_SCHEMA(__TestOpRegNoTypes).Description("Op Registration Without Types.")
                .Input("input_1", "docstr for input_1.")
                .Input("input_2", "docstr for input_2.")
                .Output("output_1", "docstr for output_1.");
            const OperatorSchema* opSchema;
            const OpSignature* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestOpRegNoTypes", &opSchema);
            EXPECT_TRUE(success);
            op = &(opSchema->GetOpSignature());
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
            REGISTER_OPERATOR_SCHEMA(__TestOpReg).Description("Op Registration Basic Test.")
                .Input("input_1", "docstr for input_1.", "tensor(int32)")
                .Input("input_2", "docstr for input_2.", "tensor(int32)")
                .Output("output_1", "docstr for output_1.", "tensor(int32)");
            const OperatorSchema* opSchema;
            const OpSignature* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestOpReg", &opSchema);
            EXPECT_TRUE(success);
            op = &(opSchema->GetOpSignature());
            EXPECT_EQ(op->GetInputs().size(), 2);
            EXPECT_EQ(op->GetInputs()[0].GetName(), "input_1");
            EXPECT_EQ(op->GetInputs()[0].GetTypes().size(), 1);
            EXPECT_EQ(**op->GetInputs()[0].GetTypes().find(Utils::OpUtils::ToType("tensor(int32)")), "tensor(int32)");
            EXPECT_EQ(op->GetInputs()[1].GetName(), "input_2");
            EXPECT_EQ(op->GetInputs()[1].GetTypes().size(), 1);
            EXPECT_EQ(**op->GetInputs()[1].GetTypes().find(Utils::OpUtils::ToType("tensor(int32)")), "tensor(int32)");
            EXPECT_EQ(op->GetOutputs().size(), 1);
            EXPECT_EQ(op->GetOutputs()[0].GetName(), "output_1");
            EXPECT_EQ(op->GetOutputs()[0].GetTypes().size(), 1);
            EXPECT_EQ(**op->GetOutputs()[0].GetTypes().find(Utils::OpUtils::ToType("tensor(int32)")), "tensor(int32)");
        }

        TEST(OpRegistrationTest, TypeConstraintTest)
        {
            REGISTER_OPERATOR_SCHEMA(__TestTypeConstraint).Description("Op with Type Constraint.")
                .Input("input_1", "docstr for input_1.", "T")
                .Input("input_2", "docstr for input_2.", "T")
                .Output("output_1", "docstr for output_1.", "T")
                .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrain input and output types to floats.");
            const OperatorSchema* opSchema;
            const OpSignature* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestTypeConstraint", &opSchema);
            EXPECT_TRUE(success);
            op = &(opSchema->GetOpSignature());
            EXPECT_EQ(op->GetInputs().size(), 2);
            EXPECT_EQ(op->GetInputs()[0].GetName(), "input_1");
            EXPECT_EQ(op->GetInputs()[0].GetTypes().size(), 3);
            EXPECT_EQ(**op->GetInputs()[0].GetTypes().find(Utils::OpUtils::ToType("tensor(float16)")), "tensor(float16)");
            EXPECT_EQ(**op->GetInputs()[0].GetTypes().find(Utils::OpUtils::ToType("tensor(float)")), "tensor(float)");
            EXPECT_EQ(**op->GetInputs()[0].GetTypes().find(Utils::OpUtils::ToType("tensor(double)")), "tensor(double)");

            EXPECT_EQ(op->GetInputs()[1].GetName(), "input_2");
            EXPECT_EQ(op->GetInputs()[1].GetTypes().size(), 3);
            EXPECT_EQ(**op->GetInputs()[1].GetTypes().find(Utils::OpUtils::ToType("tensor(float16)")), "tensor(float16)");
            EXPECT_EQ(**op->GetInputs()[1].GetTypes().find(Utils::OpUtils::ToType("tensor(float)")), "tensor(float)");
            EXPECT_EQ(**op->GetInputs()[1].GetTypes().find(Utils::OpUtils::ToType("tensor(double)")), "tensor(double)");

            EXPECT_EQ(op->GetOutputs().size(), 1);
            EXPECT_EQ(op->GetOutputs()[0].GetName(), "output_1");
            EXPECT_EQ(op->GetOutputs()[0].GetTypes().size(), 3);
            EXPECT_EQ(**op->GetOutputs()[0].GetTypes().find(Utils::OpUtils::ToType("tensor(float16)")), "tensor(float16)");
            EXPECT_EQ(**op->GetOutputs()[0].GetTypes().find(Utils::OpUtils::ToType("tensor(float)")), "tensor(float)");
            EXPECT_EQ(**op->GetOutputs()[0].GetTypes().find(Utils::OpUtils::ToType("tensor(double)")), "tensor(double)");
        }

        TEST(OpRegistrationTest, AttributeTest)
        {
            REGISTER_OPERATOR_SCHEMA(__TestAttr).Description("Op with attributes.")
                .Attr("my_attr_int", "attr with INT type", AttrType::AttributeProto_AttributeType_INT)
                .Attr("my_attr_float", "attr with FLOAT type", AttrType::AttributeProto_AttributeType_FLOAT)
                .Attr("my_attr_string", "attr with STRING type", AttrType::AttributeProto_AttributeType_STRING);
            const OperatorSchema* opSchema;
            const OpSignature* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestAttr", &opSchema);
            EXPECT_TRUE(success);
            op = &(opSchema->GetOpSignature());

            std::vector<std::string> expected_strings = { "my_attr_int", "my_attr_float", "my_attr_string" };
            std::vector<AttrType> expected_types = { AttrType::AttributeProto_AttributeType_INT, AttrType::AttributeProto_AttributeType_FLOAT, AttrType::AttributeProto_AttributeType_STRING };

            size_t size = op->GetAttributes().size();
            EXPECT_EQ(size, 3);
            for (size_t i = 0; i < size; i++)
            {
                EXPECT_EQ(op->GetAttributes()[i].GetName(), expected_strings[i]);
                EXPECT_EQ(op->GetAttributes()[i].GetType(), expected_types[i]);
            }
        }

        TEST(OpRegistrationTest, AttributeDefaultValueTest)
        {
            REGISTER_OPERATOR_SCHEMA(__TestAttrDefaultValue).Description("Op with attributes that have default values")
                .Attr("my_attr_int", "attr with default value of 99.", AttrType::AttributeProto_AttributeType_INT, int64_t(99))
                .Attr("my_attr_float", "attr with default value of 0.99.", AttrType::AttributeProto_AttributeType_FLOAT, float(0.99))
                .Attr("my_attr_string", "attr with default value of \"99\".", AttrType::AttributeProto_AttributeType_STRING, std::string("99"));
            const OperatorSchema* opSchema;
            const OpSignature* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestAttrDefaultValue", &opSchema);
            EXPECT_TRUE(success);
            op = &(opSchema->GetOpSignature());
            EXPECT_EQ(op->GetAttributes().size(), 3);

            EXPECT_EQ(op->GetAttributes()[0].GetName(), "my_attr_int");
            EXPECT_EQ(op->GetAttributes()[0].GetType(), AttrType::AttributeProto_AttributeType_INT);
            const AttributeProto* a1;
            EXPECT_TRUE(op->GetAttributes()[0].HasDefaultValue(&a1));
            EXPECT_EQ(a1->name(), "my_attr_int");
            EXPECT_TRUE(a1->has_i());
            EXPECT_EQ(a1->i(), 99LL);

            EXPECT_EQ(op->GetAttributes()[1].GetName(), "my_attr_float");
            EXPECT_EQ(op->GetAttributes()[1].GetType(), AttrType::AttributeProto_AttributeType_FLOAT);
            const AttributeProto* a2;
            EXPECT_TRUE(op->GetAttributes()[1].HasDefaultValue(&a2));
            EXPECT_EQ(a2->name(), "my_attr_float");
            EXPECT_TRUE(a2->has_f());
            EXPECT_EQ(a2->f(), 0.99f);

            EXPECT_EQ(op->GetAttributes()[2].GetName(), "my_attr_string");
            EXPECT_EQ(op->GetAttributes()[2].GetType(), AttrType::AttributeProto_AttributeType_STRING);
            const AttributeProto* a3;
            EXPECT_TRUE(op->GetAttributes()[2].HasDefaultValue(&a3));
            EXPECT_EQ(a3->name(), "my_attr_string");
            EXPECT_TRUE(a3->has_s());
            EXPECT_EQ(a3->s(), "99");
        }

        TEST(OpRegistrationTest, AttributeDefaultValueListTest)
        {
            REGISTER_OPERATOR_SCHEMA(__TestAttrDefaultValueList).Description("Op with attributes that have default list of values.")
                .Attr("my_attr_ints", "attr with default value of [98, 99, 100].", AttrType::AttributeProto_AttributeType_INTS, std::vector<int64_t> {int64_t(98), int64_t(99), int64_t(100)})
                .Attr("my_attr_floats", "attr with default value of [0.98, 0.99, 1.00].", AttrType::AttributeProto_AttributeType_FLOATS, std::vector<float> {float(0.98), float(0.99), float(1.00)})
                .Attr("my_attr_strings", "attr with default value of [\"98\", \"99\", \"100\"].", AttrType::AttributeProto_AttributeType_STRINGS, std::vector<std::string> {"98", "99", "100"});
            const OperatorSchema* opSchema;
            const OpSignature* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("__TestAttrDefaultValueList", &opSchema);
            EXPECT_TRUE(success);
            op = &(opSchema->GetOpSignature());
            EXPECT_EQ(op->GetAttributes().size(), 3);

            EXPECT_EQ(op->GetAttributes()[0].GetName(), "my_attr_ints");
            EXPECT_EQ(op->GetAttributes()[0].GetType(), AttrType::AttributeProto_AttributeType_INTS);
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
            EXPECT_EQ(op->GetAttributes()[1].GetType(), AttrType::AttributeProto_AttributeType_FLOATS);
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
            EXPECT_EQ(op->GetAttributes()[2].GetType(), AttrType::AttributeProto_AttributeType_STRINGS);
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

        TEST(TestONNXReg, VerifyRegistration)
        {
            const OperatorSchema* opSchema;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("Add", &opSchema);
            EXPECT_TRUE(success);
            success = OperatorSchemaRegistry::Get()->TryGetOp("Conv", &opSchema);
            EXPECT_TRUE(success);
        }
    }
}

#include <iostream>
#include "gtest/gtest.h"
#include "core/graph/op.h"
#include "core/graph/utils.h"
#include "core/protobuf/graph.pb.h"

namespace LotusIR
{
    namespace Test
    {
        TEST(OpRegistrationTest, LinearOp)
        {
            const OperatorSchema* opSchema;
            const OpSignature* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("Linear", &opSchema);
            EXPECT_TRUE(success);
            op = &(opSchema->GetOpSignature());
            size_t input_size = op->GetInputs().size();
            EXPECT_EQ(input_size, 1);
            EXPECT_EQ(op->GetInputs()[0].GetTypes(), op->GetOutputs()[0].GetTypes());
            size_t attr_size = op->GetAttributes().size();
            EXPECT_EQ(attr_size, 2);
            EXPECT_EQ(op->GetAttributes()[0].GetName(), "alpha");
            EXPECT_EQ(op->GetAttributes()[0].GetType(), AttrType::FLOAT);
            EXPECT_EQ(op->GetAttributes()[1].GetName(), "beta");
            EXPECT_EQ(op->GetAttributes()[1].GetType(), AttrType::FLOAT);
        }

        TEST(OpRegistrationTest, EmbeddingOp)
        {
            const OperatorSchema* opSchema;
            const OpSignature* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("Embedding", &opSchema);
            EXPECT_TRUE(success);
            op = &(opSchema->GetOpSignature());
            size_t input_size = op->GetInputs().size();
            EXPECT_EQ(input_size, 1);
            DataTypeSet input_types, output_types;
            input_types.emplace(Utils::OpUtils::ToType("tensor(uint64)"));
            output_types.emplace(Utils::OpUtils::ToType("tensor(float16)"));
            output_types.emplace(Utils::OpUtils::ToType("tensor(float)"));
            output_types.emplace(Utils::OpUtils::ToType("tensor(double)"));
            EXPECT_EQ(op->GetInputs()[0].GetTypes(), input_types);
            EXPECT_EQ(op->GetOutputs()[0].GetTypes(), output_types);
            size_t attr_size = op->GetAttributes().size();
            EXPECT_EQ(attr_size, 3);
            EXPECT_EQ(op->GetAttributes()[0].GetName(), "input_dim");
            EXPECT_EQ(op->GetAttributes()[0].GetType(), AttrType::INT);
            EXPECT_EQ(op->GetAttributes()[1].GetName(), "output_dim");
            EXPECT_EQ(op->GetAttributes()[1].GetType(), AttrType::INT);
            EXPECT_EQ(op->GetAttributes()[2].GetName(), "weights");
            EXPECT_EQ(op->GetAttributes()[2].GetType(), AttrType::FLOATS);
        }
    }

}

#include <iostream>
#include "core/protobuf/graph.pb.h"
#include "gtest/gtest.h"
#include "op.h"
#include "utils.h"

namespace LotusIR
{
    namespace Test
    {
        TEST(OpRegistrationTest, ScaleOp)
        {
            const OperatorSchema* opSchema;
            const OpSignature* op;
            bool success = OperatorSchemaRegistry::Get()->TryGetOp("Scale", &opSchema);
            EXPECT_TRUE(success);
            op = &(opSchema->GetOpSignature());
            size_t input_size = op->GetInputs().size();
            EXPECT_EQ(input_size, 1);
            EXPECT_EQ(op->GetInputs()[0].GetTypes(), op->GetOutputs()[0].GetTypes());
            size_t attr_size = op->GetAttributes().size();
            EXPECT_EQ(attr_size, 1);
            EXPECT_EQ(op->GetAttributes()[0].GetName(), "scale");
            EXPECT_EQ(op->GetAttributes()[0].GetType(), AttrType::FLOAT);
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
            input_types.emplace(Utils::OpUtils::ToType( "uint64" ));
            output_types.emplace(Utils::OpUtils::ToType("float16"));
            output_types.emplace(Utils::OpUtils::ToType("float"));
            output_types.emplace(Utils::OpUtils::ToType("double"));
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

#include <iostream>
#include "gtest/gtest.h"
#include "core/graph/model.h"
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
            EXPECT_EQ(op->GetAttributes()[0].GetType(), AttrType::AttributeProto_AttributeType_FLOAT);
            EXPECT_EQ(op->GetAttributes()[1].GetName(), "beta");
            EXPECT_EQ(op->GetAttributes()[1].GetType(), AttrType::AttributeProto_AttributeType_FLOAT);
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
            EXPECT_EQ(op->GetAttributes()[0].GetType(), AttrType::AttributeProto_AttributeType_INT);
            EXPECT_EQ(op->GetAttributes()[1].GetName(), "output_dim");
            EXPECT_EQ(op->GetAttributes()[1].GetType(), AttrType::AttributeProto_AttributeType_INT);
            EXPECT_EQ(op->GetAttributes()[2].GetName(), "weights");
            EXPECT_EQ(op->GetAttributes()[2].GetType(), AttrType::AttributeProto_AttributeType_FLOATS);
        }

        TEST(FeatureVectorizerTest, TraditionalMlOpTest)
        {
            Model model("traditionalMl");
            auto graph = model.MainGraph();

            // Case: A traditional ml graph.
            //                           SouceNode
            //                              |                
            //                       node_1(CastMap)               
            //                      (tensor(float))                 
            //                             |               
            //                    node_5 (FeatureVectorizer)
            //                              |
            //                           SinkNode

            std::vector<NodeArg> inputs;
            std::vector<NodeArg> outputs;

            // Type: tensor(float)
            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

            // Type: map(int64,float);
            TypeProto map_int64_float;
            auto mapType = map_int64_float.mutable_map_type();
            mapType->set_key_type(TensorProto::INT64);
            auto mapValueType = mapType->mutable_value_type()->mutable_tensor_type();
            mapValueType->set_elem_type(TensorProto::FLOAT);
            mapValueType->mutable_shape();

            NodeArg inputArg1("node_1_in_1", &map_int64_float);
            inputs.clear();
            inputs.push_back(inputArg1);
            NodeArg outputArg1("node_1_out_1", &tensor_float);
            outputs.clear();
            outputs.push_back(outputArg1);
            graph->AddNode("node_1", "CastMap", "node 1", inputs, outputs);

            inputs.clear();
            inputs.push_back(outputArg1);

            NodeArg outputArg4("node_4_out_1", &tensor_float);
            outputs.clear();
            outputs.push_back(outputArg4);
            graph->AddNode("node_4", "FeatureVectorizer", "node 4", inputs, { 1 }, outputs);
            auto status = graph->Resolve();
            EXPECT_TRUE(status.Ok());
        }
    }
}

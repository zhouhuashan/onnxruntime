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
            //                                     SouceNode
            //                  /                     |                              \                                \
            //           node_1                     node_2                             node_3                          node_4
            // (Variable_FeatureVectorizer)   (Variable_FeatureVectorizer)      (Variable_FeatureVectorizer)      (Variable_FeatureVectorizer)
            //                  \                     |                              /                                /
            //                                      node_5 (FeatureVectorizer)
            //                                        |
            //                                     SinkNode
            REGISTER_OPERATOR_SCHEMA(Variable_FeatureVectorizer).Description("Input variable.")
                .Input("input_1", "docstr for input_1.", "T")
                .Output("output_1", "docstr for output_1.", "T")
                .TypeConstraint("T", { "record(name:string, value:float)",
                    "record(name:string, value:seq(float))",
                    "record(name:string, value:map(int64, float))",
                    "record(name:string, value:tensor(float))" },
                    "input/output types");

            std::vector<NodeArg> inputs;
            std::vector<NodeArg> outputs;

            // Type: record(name:string, value:float);
            TypeProto record_float;
            auto featureName = record_float.mutable_record_type()->add_field();
            featureName->set_name("name");
            auto scalarTensor = featureName->mutable_type()->mutable_tensor_type();
            scalarTensor->set_elem_type(TensorProto_DataType_STRING);
            scalarTensor->mutable_shape();
            auto featureValue = record_float.mutable_record_type()->add_field();
            featureValue->set_name("value");
            scalarTensor = featureValue->mutable_type()->mutable_tensor_type();
            scalarTensor->set_elem_type(TensorProto_DataType_FLOAT);
            scalarTensor->mutable_shape();

            NodeArg inputArg("node_1_in_1", &record_float);
            inputs.push_back(inputArg);
            NodeArg outputArg("node_1_out_1", &record_float);
            outputs.push_back(outputArg);
            graph->AddNode("node_1", "Variable_FeatureVectorizer", "node 1", inputs, outputs);

            // Type: record(name:string, value:seq(float));
            TypeProto record_seq_float;
            featureName = record_seq_float.mutable_record_type()->add_field();
            featureName->set_name("name");
            scalarTensor = featureName->mutable_type()->mutable_tensor_type();
            scalarTensor->set_elem_type(TensorProto_DataType_STRING);
            scalarTensor->mutable_shape();
            featureValue = record_seq_float.mutable_record_type()->add_field();
            featureValue->set_name("value");
            scalarTensor = featureValue->mutable_type()->mutable_seq_type()->mutable_elem_type()->mutable_tensor_type();
            scalarTensor->set_elem_type(TensorProto_DataType_FLOAT);
            scalarTensor->mutable_shape();

            NodeArg inputArg2("node_2_in_1", &record_seq_float);
            inputs.clear();
            inputs.push_back(inputArg2);
            NodeArg outputArg2("node_2_out_1", &record_seq_float);
            outputs.clear();
            outputs.push_back(outputArg2);
            graph->AddNode("node_2", "Variable_FeatureVectorizer", "node 2", inputs, outputs);

            // Type: record(name:string, value:map(int64,float));
            TypeProto record_map_int64_float;
            featureName = record_map_int64_float.mutable_record_type()->add_field();
            featureName->set_name("name");
            scalarTensor = featureName->mutable_type()->mutable_tensor_type();
            scalarTensor->set_elem_type(TensorProto_DataType_STRING);
            scalarTensor->mutable_shape();
            featureValue = record_map_int64_float.mutable_record_type()->add_field();
            featureValue->set_name("value");
            auto mapType = featureValue->mutable_type()->mutable_map_type();
            mapType->set_key_type(TensorProto_DataType_INT64);
            scalarTensor = mapType->mutable_value_type()->mutable_tensor_type();
            scalarTensor->set_elem_type(TensorProto_DataType_FLOAT);
            scalarTensor->mutable_shape();

            NodeArg inputArg3("node_3_in_1", &record_map_int64_float);
            inputs.clear();
            inputs.push_back(inputArg3);
            NodeArg outputArg3("node_3_out_1", &record_map_int64_float);
            outputs.clear();
            outputs.push_back(outputArg3);
            graph->AddNode("node_3", "Variable_FeatureVectorizer", "node 3", inputs, outputs);

            // Type: record(name:string,value:tensor(float));
            TypeProto record_tensor_float;
            featureName = record_tensor_float.mutable_record_type()->add_field();
            featureName->set_name("name");
            scalarTensor = featureName->mutable_type()->mutable_tensor_type();
            scalarTensor->set_elem_type(TensorProto_DataType_STRING);
            scalarTensor->mutable_shape();
            featureValue = record_tensor_float.mutable_record_type()->add_field();
            featureValue->set_name("value");
            scalarTensor = featureValue->mutable_type()->mutable_tensor_type();
            scalarTensor->set_elem_type(TensorProto_DataType_FLOAT);
            scalarTensor->mutable_shape();

            NodeArg inputArg4("node_4_in_1", &record_tensor_float);
            inputs.clear();
            inputs.push_back(inputArg4);
            NodeArg outputArg4("node_4_out_1", &record_tensor_float);
            outputs.clear();
            outputs.push_back(outputArg4);
            graph->AddNode("node_4", "Variable_FeatureVectorizer", "node 4", inputs, outputs);

            // Type: tensor(double)
            TypeProto tensor_double;
            tensor_double.mutable_tensor_type()->set_elem_type(TensorProto_DataType_DOUBLE);

            inputs.clear();
            inputs.push_back(outputArg);
            inputs.push_back(outputArg2);
            inputs.push_back(outputArg3);
            inputs.push_back(outputArg4);
            NodeArg outputArg5("node_5_out_1", &tensor_double);
            outputs.clear();
            outputs.push_back(outputArg5);
            graph->AddNode("node_5", "FeatureVectorizer", "node 5", inputs, { 4 }, outputs);
            auto status = graph->Resolve();
            EXPECT_TRUE(status.Ok());
        }
    }
}

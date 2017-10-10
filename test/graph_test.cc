#include <iostream>

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
#include "gtest/gtest.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"

namespace LotusIR
{
    namespace Test
    {
        using google::protobuf::util::MessageDifferencer;

        TEST(ResolvingGraphTest, GraphConstruction_VerifyNoDuplicateName)
        {
            Graph graph("graph_1");

            EXPECT_EQ("graph_1", graph.Name());

            std::vector<NodeArg> inputs;
            std::vector<NodeArg> outputs;

            // INT32 vector.
            TypeProto outputType;
            outputType.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            outputType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

            NodeArg outputArg("node_1_out_1", &outputType);
            outputs.push_back(outputArg);
            graph.AddNode("node_1", "Variable", "node 1.", inputs, outputs);

            // Case 1: Adding two nodes with same node name should fail.
            auto nodeWithDupName = graph.AddNode("node_1", "Variable", "node 2", inputs, outputs);
            auto status = graph.Resolve();
            EXPECT_FALSE(status.Ok());
            EXPECT_EQ("Error: two nodes with same node name (node_1).", status.ErrorMessage());
            graph.RemoveNode(nodeWithDupName->Index());

            // Case 2: Adding two nodes with same output arg name should fail.
            graph.AddNode("node_2", "Variable", "node 2", inputs, outputs);
            status = graph.Resolve();
            EXPECT_FALSE(status.Ok());
            EXPECT_EQ("Error: two output args with same name (node_1_out_1).", status.ErrorMessage());
        }

        TEST(ResolvingGraphTest, GraphConstruction_VerifyNodeAndOpMatch)
        {
            Graph graph("graph_1");

            std::vector<NodeArg> inputs;
            std::vector<NodeArg> outputs;

            // INT32 vector.
            TypeProto outputType;
            outputType.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            outputType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

            NodeArg outputArg("node_1_out_1", &outputType);
            outputs.push_back(outputArg);
            // Case: Adding node refering to non-existing operator should fail.
            graph.AddNode("node_1", "OpNotExist", "node 1", inputs, outputs);
            auto status = graph.Resolve();
            EXPECT_FALSE(status.Ok());
            EXPECT_EQ(
                "Error: the operator or function (OpNotExist) refered by node (node_1) does not exist.",
                status.ErrorMessage());
        }

        TEST(ResolvingGraphTest, GraphConstruction_CheckIsAcyclic)
        {
            REGISTER_OPERATOR_SCHEMA(Variable_Fake).Description("Input variable.")
                .Input("input_1", "docstr for input_1.", "int32")
                .Output("output_1", "docstr for output_1.", "int32");
            REGISTER_OPERATOR_SCHEMA(Add_Fake).Description("Add two integers.")
                .Input("input_1", "docstr for input_1.", "int32")
                .Input("input_2", "docstr for input_2.", "int32")
                .Output("output_1", "docstr for output_1.", "int32");
            REGISTER_OPERATOR_SCHEMA(NoOp_Fake).Description("Operator doing nothing.")
                .Input("input_1", "docstr for input_1.", "int32")
                .Output("output_1", "docstr for output_1.", "int32");

            Model model("graph_1");
            auto& graph = *(model.MainGraph());

            // Case 1: A normal graph.
            //                 SouceNode
            //                 /       \
            //  node_1 (Variable)      node_2 (Variable)
            //                 \       /
            //                 node_3 (Add)
            //                     |
            //                 node_4 (NoOp)
            //                     |
            //                  SinkNode
            std::vector<NodeArg> inputs;
            std::vector<NodeArg> outputs;

            std::unordered_map<std::string, std::pair<std::vector<NodeArg>, std::vector<NodeArg>>> expectedNodeNameToInputOutputArgs;

            TypeProto tensor_int32;
            tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

            NodeArg inputArg("node_1_in_1", &tensor_int32);
            inputs.push_back(inputArg);
            NodeArg outputArg("node_1_out_1", &tensor_int32);
            outputs.push_back(outputArg);
            expectedNodeNameToInputOutputArgs["node_1"] = { inputs, outputs };
            auto node_1 = graph.AddNode("node_1", "Variable_Fake", "node 1", inputs, outputs);

            NodeArg inputArg2("node_2_in_1", &tensor_int32);
            inputs.clear();
            inputs.push_back(inputArg2);
            NodeArg outputArg2("node_2_out_1", &tensor_int32);
            outputs.clear();
            outputs.push_back(outputArg2);
            expectedNodeNameToInputOutputArgs["node_2"] = { inputs, outputs };
            graph.AddNode("node_2", "Variable_Fake", "node 2", inputs, outputs);

            inputs.clear();
            inputs.push_back(outputArg);
            inputs.push_back(outputArg2);
            NodeArg outputArg3("node_3_out_1", &tensor_int32);
            outputs.clear();
            outputs.push_back(outputArg3);
            expectedNodeNameToInputOutputArgs["node_3"] = { inputs, outputs };
            graph.AddNode("node_3", "Add_Fake", "node 3", inputs, outputs);

            inputs.clear();
            inputs.push_back(outputArg3);
            NodeArg outputArg4("node_4_out_1", &tensor_int32);
            outputs.clear();
            outputs.push_back(outputArg4);
            expectedNodeNameToInputOutputArgs["node_4"] = { inputs, outputs };
            graph.AddNode("node_4", "NoOp_Fake", "node 4", inputs, outputs);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());

            EXPECT_TRUE(Model::Save(model, "graph_1.pb").Ok());
            std::shared_ptr<Model> model2;
            EXPECT_TRUE(Model::Load("graph_1.pb", &model2).Ok());

            auto& modelProto = model.ToProto();
            auto& modelProto2 = model2->ToProto();
            bool equalProto1And2 = MessageDifferencer::MessageDifferencer::Equals(modelProto, modelProto2);
            std::string diff;
            if (!equalProto1And2)
            {
                MessageDifferencer d;
                d.ReportDifferencesToString(&diff);
                d.Compare(modelProto, modelProto2);
            }
            else
            {
                diff = "it's fine";
            }
            EXPECT_TRUE(equalProto1And2) << diff;

            model2.reset(new Model(modelProto2));
            // Load the model again to ensure that it's still the right thing.
            Graph* graph2 = model2->MainGraph();
            for (auto nodeIter = graph2->Nodes_begin(); nodeIter != graph2->Nodes_end(); ++nodeIter)
            {
                if (graph2->IsSourceNode((*nodeIter)->Index())
                    || graph2->IsSinkNode((*nodeIter)->Index()))
                {
                    continue;
                }
                auto nodeNameToInputOutputIter = expectedNodeNameToInputOutputArgs.find((*nodeIter)->Name());
                EXPECT_FALSE(nodeNameToInputOutputIter == expectedNodeNameToInputOutputArgs.end());

                EXPECT_EQ(nodeNameToInputOutputIter->second.first.size(), (*nodeIter)->InputDefs().size());
                for (size_t i = 0; i < nodeNameToInputOutputIter->second.first.size();++i)
                {
                    EXPECT_EQ(nodeNameToInputOutputIter->second.first[i].Name(), (*nodeIter)->InputDefs()[i].Name());
                    EXPECT_EQ(nodeNameToInputOutputIter->second.first[i].Type(), (*nodeIter)->InputDefs()[i].Type());
                }

                EXPECT_EQ(nodeNameToInputOutputIter->second.second.size(), (*nodeIter)->OutputDefs().size());
                for (size_t i = 0; i < nodeNameToInputOutputIter->second.second.size();++i)
                {
                    EXPECT_EQ(nodeNameToInputOutputIter->second.second[i].Name(), (*nodeIter)->OutputDefs()[i].Name());
                    EXPECT_EQ(nodeNameToInputOutputIter->second.second[i].Type(), (*nodeIter)->OutputDefs()[i].Type());
                }
            }

            // Case 2 : The graph is not acyclic.  node_1 -> node_3 -> node_4 -> node_1.
            node_1->Mutable_InputDefs()[0] = outputArg4;
            status = graph.Resolve();
            EXPECT_FALSE(status.Ok());
            EXPECT_EQ("Error: the graph is not acyclic.", status.ErrorMessage());
        }

        TEST(ResolvingGraphTest, GraphConstruction_TypeInference)
        {
            REGISTER_OPERATOR_SCHEMA(Variable2_Fake).Description("Input variable.")
                .Input("input_1", "docstr for input_1.", "T")
                .Output("output_1", "docstr for output_1.", "T")
                .TypeConstraint("T", { "int32","float" }, "input/output types");

            REGISTER_OPERATOR_SCHEMA(Max_Fake).Description("Add two integers.")
                .Input("input_1", "docstr for input_1.", "T")
                .Output("output_1", "docstr for output_1.", "T")
                .TypeConstraint("T", { "int32","float" }, "input/output types");

            Graph graph("graph_1");

            // Case 1: A normal graph.
            //                         SouceNode
            //                     /       |           \
			//  node_1 (Variable)    node_2 (Variable) node_3 (Variable)
            //                            \|/ (it's all 3 nodes above outputs to the one input of node_4)
            //                        node_4 (Max)
            //                             |
            //                          SinkNode
            std::vector<NodeArg> inputs;
            std::vector<NodeArg> outputs;

            TypeProto tensor_int32;
            tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

            NodeArg inputArg("node_1_in_1", &tensor_int32);
            inputs.push_back(inputArg);
            NodeArg outputArg("node_1_out_1", &tensor_int32);
            outputs.push_back(outputArg);
            graph.AddNode("node_1", "Variable2_Fake", "node 1", inputs, outputs);

            NodeArg inputArg2("node_2_in_1", &tensor_int32);
            inputs.clear();
            inputs.push_back(inputArg2);
            NodeArg outputArg2("node_2_out_1", &tensor_int32);
            outputs.clear();
            outputs.push_back(outputArg2);
            auto node_2 = graph.AddNode("node_2", "Variable2_Fake", "node 2", inputs, outputs);

            NodeArg inputArg3("node_3_in_1", &tensor_int32);
            inputs.clear();
            inputs.push_back(inputArg3);
            NodeArg outputArg3("node_3_out_1", &tensor_int32);
            outputs.clear();
            outputs.push_back(outputArg3);
            graph.AddNode("node_3", "Variable2_Fake", "node 3", inputs, outputs);

            inputs.clear();
            inputs.push_back(outputArg);
            inputs.push_back(outputArg2);
            inputs.push_back(outputArg3);
            NodeArg outputArg4("node_4_out_1", &tensor_int32);
            outputs.clear();
            outputs.push_back(outputArg4);
            graph.AddNode("node_4", "Max_Fake", "node 4", inputs, { 3 }, outputs);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());

            auto& graphProto = graph.ToGraphProto();
            EXPECT_EQ(3, graphProto.input_size());
            std::string expectedGraphInputs = " node_1_in_1, node_2_in_1, node_3_in_1";
            EXPECT_GT(expectedGraphInputs.find(graphProto.input(0).name()), 0);
            EXPECT_GT(expectedGraphInputs.find(graphProto.input(1).name()), 0);
            EXPECT_GT(expectedGraphInputs.find(graphProto.input(2).name()), 0);
            EXPECT_EQ("node_4_out_1", graphProto.output(0).name());
            EXPECT_EQ(1, graphProto.output_size());

            TypeProto tensor_float;
            tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            node_2->Mutable_InputDefs()[0] = NodeArg("node_2_in_1", &tensor_float);
            node_2->Mutable_OutputDefs()[0] = NodeArg("node_2_out_1", &tensor_float);
            status = graph.Resolve();
            EXPECT_FALSE(status.Ok());
            EXPECT_EQ("Node (node_4) has different input types (int32,float) matching to same type string (T).", status.ErrorMessage());
        }

        TEST(TestAddAttribute, AddTensorAttribute)
        {
            REGISTER_OPERATOR_SCHEMA(__Constant).Description("Constant Op.")
                .Attr(c_constantValue, "constant value", AttrType::TENSOR)
                .Output("output_1", "docstr for output_1.", "int64");
            std::vector<NodeArg> inputs;
            std::vector<NodeArg> outputs;
            Graph graph("graph_1");
            TypeProto outputType;
            outputType.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
            TypeProto::TensorShapeProto outputShape;
            outputShape.mutable_dim()->Add()->set_dim_value(1);
            outputShape.mutable_dim()->Add()->set_dim_value(3);
            *(outputType.mutable_tensor_type()->mutable_shape()) = outputShape;
            NodeArg outputArg("node_1_out_1", &outputType);
            outputs.push_back(outputArg);
            auto node_1 = graph.AddNode("node_1", "__Constant", "node 1.", inputs, outputs);
            TensorProto t;
            t.set_data_type(TensorProto_DataType_INT64);
            *(t.mutable_int64_data()->Add()) = 1;
            *(t.mutable_int64_data()->Add()) = 2;
            *(t.mutable_int64_data()->Add()) = 3;
            *(t.mutable_dims()->Add()) = 1;
            *(t.mutable_dims()->Add()) = 3;
            EXPECT_TRUE(node_1->AddAttribute(c_constantValue, t));
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());
        }

        TEST(CreateOnnxModelFromScratch, GraphConstruction_SkipTypeCheckingForOnnx)
        {
            Model model("graph_1");
            Model model_onnx("graph_1_onnx", true);
            Graph* graph = model.MainGraph();
            Graph* graphOnnx = model_onnx.MainGraph();

            EXPECT_EQ("graph_1", graph->Name());

            std::vector<NodeArg> inputs;
            std::vector<NodeArg> outputs;

            TypeProto floatTensor; // NCHW
            floatTensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
            floatTensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            floatTensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            floatTensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            floatTensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

            inputs.push_back(NodeArg("node_1_in_1", &floatTensor));
            outputs.push_back(NodeArg("node_1_out_1", &floatTensor));

            floatTensor.mutable_tensor_type()->mutable_shape()->clear_dim();
            floatTensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            inputs.push_back(NodeArg("node_1_in_2", &floatTensor));
            inputs.push_back(NodeArg("node_1_in_3", &floatTensor));
            inputs.push_back(NodeArg("node_1_in_4", &floatTensor));
            inputs.push_back(NodeArg("node_1_in_5", &floatTensor));

            auto node_1 = graph->AddNode("node_1", "BatchNormalization", "node 1.", inputs, outputs);
            EXPECT_TRUE(nullptr != node_1);
            auto node_onnx = graphOnnx->AddNode("node_1", "BatchNormalization", "node 1.", inputs, outputs);
            EXPECT_TRUE(nullptr != node_onnx);
            auto status = graph->Resolve();
            EXPECT_FALSE(status.Ok());
            EXPECT_EQ("Error: node (node_1)'s number of outputs does not match its operator (BatchNormalization) specification.", status.ErrorMessage());
            status = graphOnnx->Resolve();
            EXPECT_TRUE(status.Ok());
        }
    }
}

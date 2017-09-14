#include <iostream>
#include "gtest/gtest.h"
#include "graph.h"

namespace LotusIR
{
    namespace Test
    {
        TEST(GraphConstruction_VerifyNoDuplicateName, ResolvingGraphTest)
        {
            Graph graph("graph_1", 1, 1, "tag_1");

            EXPECT_EQ(1, graph.IrVersion());
            EXPECT_EQ("graph_1", graph.Name());

            std::vector<NodeArg> inputs;
            std::vector<NodeArg> outputs;

            TypeProto outputType;
            outputType.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            TensorShapeProto outputShape;
            outputShape.add_dim()->set_dim_value(1);

            NodeArg outputArg("node_1_out_1", outputType, outputShape);
            outputs.push_back(outputArg);
            auto node_1 = graph.AddNode("node_1", "Variable", inputs, outputs);

            // Case 1: Adding two nodes with same node name should fail.
            auto nodeWithDupName = graph.AddNode("node_1", "Variable", inputs, outputs);
            auto status = graph.Resolve();
            EXPECT_FALSE(status.Ok());
            EXPECT_EQ("Error: two nodes with same node name (node_1).", status.ErrorMsg());
            graph.RemoveNode(nodeWithDupName->Index());

            // Case 2: Adding two nodes with same output arg name should fail.
            auto nodeWithDupOutputArgName = graph.AddNode("node_2", "Variable", inputs, outputs);
            status = graph.Resolve();
            EXPECT_FALSE(status.Ok());
            EXPECT_EQ("Error: two output args with same name (node_1_out_1).", status.ErrorMsg());
        }

        TEST(GraphConstruction_VerifyNodeAndOpMatch, ResolvingGraphTest)
        {
            Graph graph("graph_1", 1, 1, "tag_1");

            std::vector<NodeArg> inputs;
            std::vector<NodeArg> outputs;

            TypeProto outputType;
            outputType.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            TensorShapeProto outputShape;
            outputShape.add_dim()->set_dim_value(1);

            NodeArg outputArg("node_1_out_1", outputType, outputShape);
            outputs.push_back(outputArg);
            // Case: Adding node refering to non-existing operator should fail.
            auto nodeWithOpNotExist = graph.AddNode("node_1", "OpNotExist", inputs, outputs);
            auto status = graph.Resolve();
            EXPECT_FALSE(status.Ok());
            EXPECT_EQ(
                "Error: the operator or function (OpNotExist) refered by node (node_1) does not exist.",
                status.ErrorMsg());
        }

        TEST(GraphConstruction_CheckIsAcyclic, ResolvingGraphTest)
        {
            Graph graph("graph_1", 1, 1, "tag_1");

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

            TypeProto outputType;
            outputType.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
            TensorShapeProto outputShape;
            outputShape.add_dim()->set_dim_value(1);

            NodeArg outputArg("node_1_out_1", outputType, outputShape);
            outputs.push_back(outputArg);
            auto node_1 = graph.AddNode("node_1", "Variable", inputs, outputs);

            NodeArg outputArg2("node_2_out_1", outputType, outputShape);
            outputs.clear();
            outputs.push_back(outputArg2);
            auto node_2 = graph.AddNode("node_2", "Variable", inputs, outputs);

            inputs.push_back(outputArg);
            inputs.push_back(outputArg2);
            NodeArg outputArg3("node_3_out_1", outputType, outputShape);
            outputs.clear();
            outputs.push_back(outputArg3);
            auto node_3 = graph.AddNode("node_3", "Add", inputs, outputs);

            inputs.clear();
            inputs.push_back(outputArg3);
            NodeArg outputArg4("node_4_out_1", outputType, outputShape);
            outputs.clear();
            outputs.push_back(outputArg4);
            auto node_4 = graph.AddNode("node_4", "NoOp", inputs, outputs);
            auto status = graph.Resolve();
            EXPECT_TRUE(status.Ok());

            // Case 2 : The graph is not acyclic.  node_1 -> node_3 -> node_4 -> node_1.
            // TODO: "Variable", "Add", "NoOp" operators should be registered, otherwise,
            // error of referring non-existing op will be returned firstly.
            node_1->Mutable_InputDefs().push_back(outputArg4);
            status = graph.Resolve();
            EXPECT_FALSE(status.Ok());
            EXPECT_EQ("Error: the graph is not acyclic.", status.ErrorMsg());
        }
    }
}

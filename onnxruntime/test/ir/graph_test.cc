// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#ifdef _MSC_VER
#pragma warning(push)
// 'identifier' : unreferenced formal parameter
#pragma warning(disable : 4100)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable : 4800)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "google/protobuf/util/message_differencer.h"
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "gtest/gtest.h"
#include "core/graph/function_container.h"

#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

#define OPERATOR_SCHEMA UNUSED ONNX_OPERATOR_SCHEMA

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {
using google::protobuf::util::MessageDifferencer;

TEST(GraphTraversalTest, ReverseDFS) {
  OPERATOR_SCHEMA(Variable_DFS)
      .SetDoc("Input variable.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  OPERATOR_SCHEMA(Add_DFS)
      .SetDoc("Add two integers.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Input(1, "input_2", "docstr for input_2.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  OPERATOR_SCHEMA(NoOp_DFS)
      .SetDoc("Operator doing nothing.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");

  Model model("graph_1");
  auto& graph = model.MainGraph();

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
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  inputs.push_back(&input_arg);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Variable_DFS", "node 1", inputs, outputs);

  auto& input_arg2 = graph.GetOrCreateNodeArg("node_2_in_1", &tensor_int32);
  inputs.clear();
  inputs.push_back(&input_arg2);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg2);
  graph.AddNode("node_2", "Variable_DFS", "node 2", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&output_arg2);
  auto& output_arg3 = graph.GetOrCreateNodeArg("node_3_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg3);
  graph.AddNode("node_3", "Add_DFS", "node 3", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg3);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg4);
  graph.AddNode("node_4", "NoOp_DFS", "node 4", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::vector<const Node*> from;
  from.push_back(graph.SinkNode());

  std::vector<std::string> enter_leave_sequence;

  struct NodeCompareName {
    bool operator()(const Node* n1, const Node* n2) const {
      return n1->Name() < n2->Name();
    }
  };

  graph.ReverseDFSFrom(from,
                       [&enter_leave_sequence](const Node* n) {
                         std::string s("enter:");
                         s += n->Name();
                         enter_leave_sequence.push_back(s);
                       },
                       [&enter_leave_sequence](const Node* n) {
                         std::string s("leave:");
                         s += n->Name();
                         enter_leave_sequence.push_back(s);
                       },
                       NodeCompareName());

  /*for (size_t i = 0; i < enter_leave_sequence.size(); i++) {
        printf("%s\n", enter_leave_sequence[i].c_str());
    }*/

  EXPECT_EQ(enter_leave_sequence.size(), 12);
  EXPECT_EQ("enter:_Graph_Sink", enter_leave_sequence[0]);
  EXPECT_EQ("enter:node_4", enter_leave_sequence[1]);
  EXPECT_EQ("enter:node_3", enter_leave_sequence[2]);
  EXPECT_EQ("enter:node_2", enter_leave_sequence[3]);
  EXPECT_EQ("enter:_Graph_Source", enter_leave_sequence[4]);
  EXPECT_EQ("leave:_Graph_Source", enter_leave_sequence[5]);
  EXPECT_EQ("leave:node_2", enter_leave_sequence[6]);
  EXPECT_EQ("enter:node_1", enter_leave_sequence[7]);
  EXPECT_EQ("leave:node_1", enter_leave_sequence[8]);
  EXPECT_EQ("leave:node_3", enter_leave_sequence[9]);
  EXPECT_EQ("leave:node_4", enter_leave_sequence[10]);
  EXPECT_EQ("leave:_Graph_Sink", enter_leave_sequence[11]);
}

TEST(ResolvingGraphTest, GraphConstruction_VerifyNoDuplicateName) {
  Model model("graph_1");
  auto& graph = model.MainGraph();

  EXPECT_EQ("graph_1", graph.Name());

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // INT32 vector.
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Variable", "node 1.", inputs, outputs);

  // Case 1: Adding two nodes with same node name should fail.
  auto node_with_dup_name = graph.AddNode("node_1", "Variable", "node 2", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ("Error: two nodes with same node name (node_1).", status.ErrorMessage());
  graph.RemoveNode(node_with_dup_name->Index());

  // Case 2: Adding two nodes with same output arg name should fail.
  graph.AddNode("node_2", "Variable", "node 2", inputs, outputs);
  status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  bool duplicate_error_found = status.ErrorMessage().find("Duplicate") != std::string::npos;
  EXPECT_TRUE(duplicate_error_found);
}

TEST(ResolvingGraphTest, GraphConstruction_VerifyNodeAndOpMatch) {
  Model model("graph_1");
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // INT32 vector.
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  // Case: Adding node refering to non-existing operator should fail.
  graph.AddNode("node_1", "OpNotExist", "node 1", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(0, status.ErrorMessage().find_first_of("No Schema registered for OpNotExist"));
}

TEST(ResolvingGraphTest, GraphConstruction_CheckIsAcyclic) {
  OPERATOR_SCHEMA(Variable_Fake)
      .SetDoc("Input variable.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  OPERATOR_SCHEMA(Add_Fake)
      .SetDoc("Add two integers.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Input(1, "input_2", "docstr for input_2.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  OPERATOR_SCHEMA(NoOp_Fake)
      .SetDoc("Operator doing nothing.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");

  Model model("graph_1");
  auto& graph = model.MainGraph();

  // A normal graph.
  //                 SouceNode
  //                 /       \
  //    node_1 (Variable)  node_2 (Variable)
  //                 \       /
  //                 node_3 (Add)
  //                     |
  //                 node_4 (NoOp)
  //                     |
  //                  SinkNode
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  std::unordered_map<std::string, std::pair<std::vector<NodeArg*>, std::vector<NodeArg*>>>
      expected_node_name_to_input_output_args;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg1 = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  inputs.push_back(&input_arg1);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  outputs.push_back(&output_arg1);
  expected_node_name_to_input_output_args["node_1"] = {inputs, outputs};
  graph.AddNode("node_1", "Variable_Fake", "node 1", inputs, outputs);

  auto& input_arg2 = graph.GetOrCreateNodeArg("node_2_in_1", &tensor_int32);
  inputs.clear();
  inputs.push_back(&input_arg2);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg2);
  expected_node_name_to_input_output_args["node_2"] = {inputs, outputs};
  graph.AddNode("node_2", "Variable_Fake", "node 2", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg1);
  inputs.push_back(&output_arg2);
  auto& output_arg3 = graph.GetOrCreateNodeArg("node_3_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg3);
  expected_node_name_to_input_output_args["node_3"] = {inputs, outputs};
  graph.AddNode("node_3", "Add_Fake", "node 3", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg3);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg4);
  expected_node_name_to_input_output_args["node_4"] = {inputs, outputs};
  graph.AddNode("node_4", "NoOp_Fake", "node 4", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  EXPECT_TRUE(Model::Save(model, "graph_1.pb").IsOK());
  std::shared_ptr<Model> model2;
  EXPECT_TRUE(Model::Load("graph_1.pb", model2).IsOK());

  auto model_proto = model.ToProto();
  auto model_proto2 = model2->ToProto();
  bool equal_proto_1_and_2 = MessageDifferencer::MessageDifferencer::Equals(model_proto, model_proto2);
  std::string diff;
  if (!equal_proto_1_and_2) {
    MessageDifferencer d;
    d.ReportDifferencesToString(&diff);
    d.Compare(model_proto, model_proto2);
  } else {
    diff = "it's fine";
  }
  EXPECT_TRUE(equal_proto_1_and_2) << diff;

  // Load the model again to ensure that it's still the right thing.
  //EXPECT_EQ(Model::Load(model_proto2, &model2), Status::OK());
  model2.reset(new Model(model_proto2));
  Graph& graph2 = model2->MainGraph();
  for (auto& node : graph2.Nodes()) {
    if (graph2.IsSourceNode(node.Index()) || graph2.IsSinkNode(node.Index())) {
      continue;
    }
    auto node_name_to_input_output_iter = expected_node_name_to_input_output_args.find(node.Name());
    EXPECT_FALSE(node_name_to_input_output_iter == expected_node_name_to_input_output_args.end());

    EXPECT_EQ(node_name_to_input_output_iter->second.first.size(), node.InputDefs().size());
    for (size_t i = 0; i < node_name_to_input_output_iter->second.first.size(); ++i) {
      EXPECT_EQ(node_name_to_input_output_iter->second.first[i]->Name(), node.InputDefs()[i]->Name());
      EXPECT_EQ(node_name_to_input_output_iter->second.first[i]->Type(), node.InputDefs()[i]->Type());
    }

    EXPECT_EQ(node_name_to_input_output_iter->second.second.size(), node.OutputDefs().size());
    for (size_t i = 0; i < node_name_to_input_output_iter->second.second.size(); ++i) {
      EXPECT_EQ(node_name_to_input_output_iter->second.second[i]->Name(), node.OutputDefs()[i]->Name());
      EXPECT_EQ(node_name_to_input_output_iter->second.second[i]->Type(), node.OutputDefs()[i]->Type());
    }
  }
}

TEST(ResolvingGraphTest, GraphConstruction_CheckIsNotAcyclic) {
  // A cyclic graph
  //                 SouceNode
  //                     |
  //             --> node_1 (Add)
  //            ^        |
  //            | <- node_2 (NoOp)

  OPERATOR_SCHEMA(Add_Fake)
      .SetDoc("Add two integers.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Input(1, "input_2", "docstr for input_2.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  OPERATOR_SCHEMA(NoOp_Fake)
      .SetDoc("Operator doing nothing.")
      .Input(0, "input_1", "docstr for input_1.", "tensor(int32)")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  Model model("graph_1");
  auto& graph = model.MainGraph();
  auto& input_arg1 = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  auto& output_arg1 = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  inputs.push_back(&input_arg1);
  inputs.push_back(&output_arg2);
  outputs.push_back(&output_arg1);
  graph.AddNode("node_1", "Add_Fake", "node 1", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg1);
  outputs.clear();
  outputs.push_back(&output_arg2);
  graph.AddNode("node_2", "NoOp_Fake", "node 2", inputs, outputs);

  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ("Error: the graph is not acyclic.", status.ErrorMessage());
}

TEST(ResolvingGraphTest, GraphConstruction_OnlyInitializer) {
  onnxruntime::Model model("graph");
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TensorProto weight;
  weight.add_dims(1);
  weight.set_data_type(TensorProto_DataType_STRING);
  weight.add_string_data("test");
  weight.set_name("node_1_in_2");
  graph.AddInitializedTensor(weight);

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(ResolvingGraphTest, GraphConstruction_TypeInference) {
  OPERATOR_SCHEMA(Variable2_Fake)
      .SetDoc("Input variable.")
      .Input(0, "input_1", "docstr for input_1.", "T")
      .Output(0, "output_1", "docstr for output_1.", "T")
      .TypeConstraint("T", {"tensor(int32)", "tensor(float)"}, "input/output types");

  OPERATOR_SCHEMA(Max_Fake)
      .SetDoc("Add two integers.")
      .Input(0, "input_1", "docstr for input_1.", "T")
      .Input(1, "input_2", "docstr for input_2.", "T")
      .Input(2, "input_3", "docstr for input_3.", "T")
      .Output(0, "output_1", "docstr for output_1.", "T")
      .TypeConstraint("T", {"tensor(int32)", "tensor(float)"}, "input/output types");

  Model model("graph_1");
  auto& graph = model.MainGraph();

  // Case 1: A normal graph.
  //                         SourceNode
  //                   /         |         \
  //  node_1 (Variable)  node_2 (Variable)  node_3 (Variable)
  //                   \         |         / (it's all 3 nodes above outputs to the one input of node_4)
  //                        node_4 (Max)
  //                             |
  //                          SinkNode
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& input_arg = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_int32);
  inputs.push_back(&input_arg);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_int32);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Variable2_Fake", "node 1", inputs, outputs);

  inputs.clear();
  inputs.push_back(&input_arg);
  auto& output_arg2 = graph.GetOrCreateNodeArg("node_2_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg2);
  graph.AddNode("node_2", "Variable2_Fake", "node 2", inputs, outputs);

  auto& input_arg3 = graph.GetOrCreateNodeArg("node_3_in_1", &tensor_int32);
  inputs.clear();
  inputs.push_back(&input_arg3);
  auto& output_arg3 = graph.GetOrCreateNodeArg("node_3_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg3);
  graph.AddNode("node_3", "Variable2_Fake", "node 3", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&output_arg2);
  inputs.push_back(&output_arg3);
  auto& output_arg4 = graph.GetOrCreateNodeArg("node_4_out_1", &tensor_int32);
  outputs.clear();
  outputs.push_back(&output_arg4);
  auto node_4 = graph.AddNode("node_4", "Max_Fake", "node 4", inputs, outputs);
  EXPECT_NE(node_4, nullptr);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::unordered_set<std::string> expected_graph_inputs = {"node_1_in_1", "node_3_in_1"};
  EXPECT_EQ(2, graph.GetInputs().size());
  for (auto& graph_input : graph.GetInputs()) {
    EXPECT_TRUE(expected_graph_inputs.find(graph_input->Name()) != expected_graph_inputs.end());
  }
  EXPECT_EQ(1, graph.GetOutputs().size());
  EXPECT_EQ("node_4_out_1", graph.GetOutputs()[0]->Name());
  EXPECT_EQ(2, graph.GetInputs().size());

  EXPECT_TRUE(Model::Save(model, "model_x.pb").IsOK());
  std::shared_ptr<Model> loaded_model;
  EXPECT_TRUE(Model::Load("model_x.pb", loaded_model).IsOK());
  EXPECT_EQ(2, loaded_model->MainGraph().GetInputs().size());

  auto& graph_proto = graph.ToGraphProto();
  EXPECT_EQ(2, graph_proto.input_size());
  for (auto& graphProtoInput : graph_proto.input()) {
    EXPECT_TRUE(expected_graph_inputs.find(graphProtoInput.name()) != expected_graph_inputs.end());
  }
  EXPECT_EQ(1, graph_proto.output_size());
  EXPECT_EQ("node_4_out_1", graph_proto.output(0).name());
}

TEST(TestAddAttribute, AddTensorAttribute) {
  OPERATOR_SCHEMA(__Constant)
      .SetDoc("Constant Op.")
      .Attr(kConstantValue, "constant value", AttrType::AttributeProto_AttributeType_TENSOR)
      .Output(0, "output_1", "docstr for output_1.", "tensor(int64)");
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  Model model("graph_1");
  auto& graph = model.MainGraph();
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  TensorShapeProto output_shape;
  output_shape.mutable_dim()->Add()->set_dim_value(1);
  output_shape.mutable_dim()->Add()->set_dim_value(3);
  *(output_type.mutable_tensor_type()->mutable_shape()) = output_shape;
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  auto node_1 = graph.AddNode("node_1", "__Constant", "node 1.", inputs, outputs);
  TensorProto t;
  t.set_data_type(TensorProto_DataType_INT64);
  *(t.mutable_int64_data()->Add()) = 1;
  *(t.mutable_int64_data()->Add()) = 2;
  *(t.mutable_int64_data()->Add()) = 3;
  *(t.mutable_dims()->Add()) = 1;
  *(t.mutable_dims()->Add()) = 3;
  node_1->AddAttribute(kConstantValue, t);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

void AddAttribute(onnxruntime::Node* p_node, const std::string& attr_name, int64_t attr_value) {
  AttributeProto attr;
  attr.set_name(attr_name);
  attr.set_type(AttributeProto_AttributeType_INT);
  attr.set_i(attr_value);
  p_node->AddAttribute(attr_name, attr);
}

void AddAttribute(onnxruntime::Node* p_node, const std::string& attr_name, std::initializer_list<int64_t> attr_value) {
  AttributeProto attr;
  attr.set_name(attr_name);
  attr.set_type(AttributeProto_AttributeType_INTS);
  for (auto v : attr_value) {
    attr.add_ints(v);
  }
  p_node->AddAttribute(attr_name, attr);
}

// Test that output type can be inferred for ops with a type-attribute
TEST(TypeInferenceTest, TypeAttribute) {
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  Model model("graph_1");
  auto& graph = model.MainGraph();
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", nullptr);
  outputs.push_back(&output_arg);
  auto node_1 = graph.AddNode("node_1", "RandomNormal", "node 1.", inputs, outputs);
  AddAttribute(node_1, "dtype", TensorProto_DataType_FLOAT);
  AddAttribute(node_1, "shape", {2, 3});
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

void CheckTensorEltType(const TypeProto* ptype, TensorProto_DataType elt_type) {
  EXPECT_NE(ptype, nullptr);
  EXPECT_TRUE(ptype->has_tensor_type());
  EXPECT_TRUE(ptype->tensor_type().has_elem_type());
  EXPECT_EQ(ptype->tensor_type().elem_type(), elt_type);
}

// Test that output type can be inferred for ops with variadic outputs
TEST(TypeInferenceTest, VariadicOutput) {
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  TypeProto tensor_type;
  tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  Model model("graph_1");
  auto& graph = model.MainGraph();
  auto& X = graph.GetOrCreateNodeArg("X", &tensor_type);
  inputs.push_back(&X);
  auto& Y = graph.GetOrCreateNodeArg("Y", nullptr);
  outputs.push_back(&Y);
  auto& Z = graph.GetOrCreateNodeArg("Z", nullptr);
  outputs.push_back(&Z);
  graph.AddNode("node_1", "Split", "node 1.", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  CheckTensorEltType(Y.TypeAsProto(), TensorProto_DataType_FLOAT);
  CheckTensorEltType(Z.TypeAsProto(), TensorProto_DataType_FLOAT);
}

// Test that Graph::Resolve checks initializer value matches the type specified in graph:
TEST(TypeInferenceTest, InitializerType) {
  Model model("graph_1");
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TensorProto weight;
  weight.set_data_type(TensorProto_DataType_INT32);
  weight.add_dims(1);
  weight.add_int32_data(1);
  weight.set_name("W");
  graph.AddInitializedTensor(weight);

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  TypeProto tensor_type;
  tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto& X = graph.GetOrCreateNodeArg("W", &tensor_type);
  inputs.push_back(&X);
  auto& Y = graph.GetOrCreateNodeArg("Y", nullptr);
  outputs.push_back(&Y);
  auto& Z = graph.GetOrCreateNodeArg("Z", nullptr);
  outputs.push_back(&Z);
  graph.AddNode("node_1", "Split", "node 1.", inputs, outputs);

  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  bool type_error_found = status.ErrorMessage().find("Type Error") != std::string::npos;
  EXPECT_TRUE(type_error_found);
}

// Test that Graph::Resolve checks initializer value matches the shape specified in graph:
TEST(TypeInferenceTest, InitializerShape) {
  Model model("graph_1");
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TensorProto weight;
  weight.set_data_type(TensorProto_DataType_FLOAT);
  weight.add_dims(1);
  weight.add_float_data(1.0f);
  weight.set_name("W");
  graph.AddInitializedTensor(weight);

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  TypeProto tensor_type;
  tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  auto& X = graph.GetOrCreateNodeArg("W", &tensor_type);
  inputs.push_back(&X);
  auto& Y = graph.GetOrCreateNodeArg("Y", nullptr);
  outputs.push_back(&Y);
  auto& Z = graph.GetOrCreateNodeArg("Z", nullptr);
  outputs.push_back(&Z);
  graph.AddNode("node_1", "Split", "node 1.", inputs, outputs);

  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  bool type_error_found = status.ErrorMessage().find("Type Error") != std::string::npos;
  EXPECT_TRUE(type_error_found);
}

// Test that Graph::Resolve identifies name-duplication across initializer and node-output-arg
TEST(NameResolutionTest, DuplicateName) {
  Model model("graph_1");
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TensorProto weight;
  weight.set_data_type(TensorProto_DataType_FLOAT);
  weight.add_dims(1);
  weight.add_float_data(1.0f);
  weight.set_name("W");
  graph.AddInitializedTensor(weight);

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  TypeProto tensor_type;
  tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  auto& X = graph.GetOrCreateNodeArg("X", &tensor_type);
  inputs.push_back(&X);
  auto& Y = graph.GetOrCreateNodeArg("Y", nullptr);
  outputs.push_back(&Y);
  auto& W = graph.GetOrCreateNodeArg("W", nullptr);
  outputs.push_back(&W);
  graph.AddNode("node_1", "Split", "node 1.", inputs, outputs);

  auto status = graph.Resolve();
  EXPECT_FALSE(status.IsOK());
  bool duplicate_error_found = status.ErrorMessage().find("Duplicate") != std::string::npos;
  EXPECT_TRUE(duplicate_error_found);
}

}  // namespace test
}  // namespace onnxruntime

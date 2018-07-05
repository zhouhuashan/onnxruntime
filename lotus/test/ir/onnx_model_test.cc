#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <memory>
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "gtest/gtest.h"
#include "test/ir/node_helper.h"

using namespace Lotus;
using namespace onnx;
namespace LotusIR {
namespace Test {
// Tests that Resolve() properly clears the state of topological sorted nodes,
// inputs, outputs and valueInfo.
// Assumes the graph passed in has been previously resolved.
void TestResolve(LotusIR::Graph* p_graph) {
  const std::vector<LotusIR::NodeIndex>* nodes;
  EXPECT_TRUE(p_graph->GetNodesInTopologicalOrder(&nodes).IsOK());
  auto nodes_before = *nodes;
  auto& inputs_before = p_graph->GetInputs();
  auto& outputs_before = p_graph->GetOutputs();
  auto& value_info_before = p_graph->GetValueInfo();

  // Touch the graph to force Resolve() to recompute.
#ifdef _WIN32
  NodeTestHelper::MutableDefinitions(*p_graph->GetNode(0)).input_arg_count;
#else
  NodeTestHelper::MutableDefinitions(*p_graph->GetNode(0));
#endif
  EXPECT_TRUE(p_graph->Resolve().IsOK());

  const std::vector<LotusIR::NodeIndex>* nodes_after;
  p_graph->GetNodesInTopologicalOrder(&nodes_after);
  auto& inputs_after = p_graph->GetInputs();
  auto& outputs_after = p_graph->GetOutputs();
  auto& value_info_after = p_graph->GetValueInfo();

  // Multiple calls to Resolve() should not alter the sorted nodes,
  // inputs, outputs and valueInfo. The internal state should be
  // cleared.
  EXPECT_EQ(nodes_before, *nodes_after);
  EXPECT_EQ(inputs_before, inputs_after);
  EXPECT_EQ(outputs_before, outputs_after);
  EXPECT_EQ(value_info_before, value_info_after);
}

TEST(ONNXModelsTest, squeeze_net) {
  // NOTE: this requires the current directory to be where LotusIR_UT.exe is located
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("./testdata/squeezenet/model.onnx", model).IsOK());
  TestResolve(model->MainGraph());
#ifdef _WIN32
  // wstring version
  std::shared_ptr<Model> model2;
  ASSERT_TRUE(Model::Load(L"./testdata/squeezenet/model.onnx", model2).IsOK());
  TestResolve(model2->MainGraph());
#endif
}

#ifdef LOTUSIR_RUN_EXTERNAL_ONNX_TESTS
TEST(ONNXModelsTest, bvlc_alexnet) {
  using ::google::protobuf::io::CodedInputStream;
  using ::google::protobuf::io::FileInputStream;
  using ::google::protobuf::io::ZeroCopyInputStream;
  int fd;
  FileOpenRd("../models/test_bvlc_alexnet/model.onnx", &fd);
  ASSERT_TRUE(fd > 0);
  std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
  std::unique_ptr<CodedInputStream> coded_input(new CodedInputStream(raw_input.get()));
  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);
  ModelProto model_proto;
  bool result = model_proto.ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();
  EXPECT_TRUE(result);
  FileClose(fd);

  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("../models/test_bvlc_alexnet/model.onnx", model).IsOK());

  // Check the graph input/output/value_info should have the same size as specified in the model file.
  EXPECT_EQ(model_proto.graph().value_info_size(), model->MainGraph()->GetValueInfo().size());
  EXPECT_EQ(model_proto.graph().input_size(), model->MainGraph()->GetInputs().size() + model->MainGraph()->GetAllInitializedTensors().size());
  EXPECT_EQ(model_proto.graph().output_size(), model->MainGraph()->GetOutputs().size());
  TestResolve(model->MainGraph());
}

TEST(ONNXModelsTest, densenet121) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("../models/test_densenet121/model.onnx", model).IsOK());
  TestResolve(model->MainGraph());
}

TEST(ONNXModelsTest, inception_v1) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("../models/test_inception_v1/model.onnx", model).IsOK());
  TestResolve(model->MainGraph());
}

TEST(ONNXModelsTest, inception_v2) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("../models/test_inception_v2/model.onnx", model).IsOK());
  TestResolve(model->MainGraph());
}

TEST(ONNXModelsTest, resnet50) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("../models/test_resnet50/model.onnx", model).IsOK());
  TestResolve(model->MainGraph());
}

TEST(ONNXModelsTest, shufflenet) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("../models/test_shufflenet/model.onnx", model).IsOK());
  TestResolve(model->MainGraph());
}

TEST(ONNXModelsTest, squeezenet) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("../models/test_squeezenet/model.onnx", model).IsOK());
  TestResolve(model->MainGraph());
}

TEST(ONNXModelsTest, zfnet) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("../models/test_zfnet/model.onnx", model).IsOK());
  TestResolve(model->MainGraph());
}

TEST(ONNXModelsTest, vgg19) {
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load("../models/test_vgg19/model.onnx", model).IsOK());
  TestResolve(model->MainGraph());
}
#endif
}  // namespace Test
}  // namespace LotusIR

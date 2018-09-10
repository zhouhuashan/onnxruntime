#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/session_state.h"
#include "core/providers/cpu/controlflow/scan.h"
#include "core/framework/customregistry.h"
#include "core/graph/function_container.h"
using namespace onnx;
namespace onnxruntime {
namespace Test {

static void CreateSubgraph(Model& model) {
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // subgraph with multiple inputs and outputs to test variadic behaviour.
  // 2 inputs of 2 that are concatenated and then split into 4 outputs of 1

  // Concat node
  {
    // input of 2 x {2} tensors
    TypeProto concat_input_tensor;
    concat_input_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    concat_input_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

    for (int i = 0, num_inputs = 2; i < num_inputs; ++i) {
      auto& input_arg = graph.GetOrCreateNodeArg("concat_in_" + std::to_string(i), &concat_input_tensor);
      inputs.push_back(&input_arg);
    }

    // one output from concatenate of {4} tensor
    TypeProto concat_output_tensor;
    concat_output_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    concat_output_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);

    auto& output_arg = graph.GetOrCreateNodeArg("concat_out_1", &concat_output_tensor);
    outputs.push_back(&output_arg);

    auto* concat = graph.AddNode("concat", "Concat", "concat 2 inputs", inputs, outputs);
    concat->AddAttribute("axis", int64_t{0});
  }

  // Split node
  {
    // setup Split to run using the Concat output
    inputs = outputs;
    outputs = {};

    // split output of 4 x {1} tensors
    TypeProto split_output_tensor;
    split_output_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    split_output_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    for (int i = 0, num_outputs = 4; i < num_outputs; ++i) {
      auto& output_arg = graph.GetOrCreateNodeArg("split_out_" + std::to_string(i), &split_output_tensor);
      outputs.push_back(&output_arg);
    }

    auto* split = graph.AddNode("split", "Split", "split into 4 outputs", inputs, outputs);
    split->AddAttribute("axis", int64_t{0});
    split->AddAttribute("split", std::vector<int64_t>{1, 1, 1, 1});
  }

  auto status = graph.Resolve();

  EXPECT_EQ(status, Status::OK());
}

static KernelDefBuilder ScanKernelDef() {
  KernelDefBuilder kdb;

  kdb.SetName("Scan")
      .SetDomain(onnxruntime::kOnnxDomain)
      .Provider(kCpuExecutionProvider)
      .TypeConstraint("V", DataTypeImpl::AllTensorTypes())
      .SinceVersion(7);

  return kdb;
}

TEST(Scan, StepOne) {
  auto op_registry = std::make_shared<CustomRegistry>();
  std::vector<onnx::OpSchema> schemas{Scan::GetScanOpSchema()};
  auto kernel_def_builder = ScanKernelDef();

  op_registry->RegisterOpSet(schemas, onnxruntime::kOnnxDomain, 7, 8);
  op_registry->RegisterCustomKernel(kernel_def_builder,
                                    [](const OpKernelInfo& info) -> OpKernel* { return new Scan(info); });

  // create model that will be used to initialize subgraph. currently there's no direct way to create a Graph instance.
  Model model("StepOne");
  CreateSubgraph(model);

  auto& graph = model.MainGraph();
  auto& proto = graph.ToGraphProto();

  OpTester test("Scan");

  // make Scan operator available in this test.
  test.AddCustomOpRegistry(op_registry);

  test.AddAttribute("body", proto);
  test.AddAttribute<std::vector<int64_t>>("scan_axes", {0});

  test.AddInput<float>("input_0", {2}, {1.f, 2.f});
  test.AddInput<float>("input_1", {2}, {3.f, 4.f});
  test.AddOutput<float>("output_0", {1}, {1.f});
  test.AddOutput<float>("output_1", {1}, {2.f});
  test.AddOutput<float>("output_2", {1}, {3.f});
  test.AddOutput<float>("output_3", {1}, {4.f});

  test.Run();
}
}  // namespace Test
}  // namespace onnxruntime

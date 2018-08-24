#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/session_state.h"
#include "core/providers/cpu/controlflow/scan.h"
#include "core/framework/customregistry.h"

using namespace onnx;
namespace Lotus {
namespace Test {

static void CreateSubgraph(Model &model) {
  auto &graph = model.MainGraph();

  std::vector<NodeArg *> inputs;
  std::vector<NodeArg *> outputs;

  // dummy node with input of Tensor<float> with shape [1,3] and output of the same
  // Simple no-op using Identity for initial testing
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  tensor_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  tensor_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  auto &input_arg = graph.GetOrCreateNodeArg("node_1_in_1", &tensor_float);
  inputs.push_back(&input_arg);
  auto &output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &tensor_float);
  outputs.push_back(&output_arg);

  (void)graph.AddNode("node_1", "Identity", "node 1", inputs, outputs);

  auto status = graph.Resolve();

  EXPECT_EQ(status, Status::OK());
}

static KernelDefBuilder ScanKernelDef() {
  KernelDefBuilder kdb;

  kdb.SetName("Scan")
      .SetDomain(LotusIR::kOnnxDomain)
      .Provider(kCpuExecutionProvider)
      .TypeConstraint("V", DataTypeImpl::AllTensorTypes())
      .SinceVersion(7);

  return kdb;
}

TEST(Scan, StepOne) {
  auto op_registry = std::make_shared<CustomRegistry>();
  std::vector<onnx::OpSchema> schemas{Scan::GetScanOpSchema()};
  auto kernel_def_builder = ScanKernelDef();

  op_registry->RegisterOpSet(schemas, LotusIR::kOnnxDomain, 7, 8);
  op_registry->RegisterCustomKernel(kernel_def_builder,
                                    [](const OpKernelInfo &info) -> OpKernel * { return new Scan(info); });

  // create model that will be used to initialize subgraph. currently there's no direct way to create a Graph instance.
  Model model("StepOne");
  CreateSubgraph(model);

  auto &graph = model.MainGraph();
  auto &proto = graph.ToGraphProto();

  OpTester test("Scan");

  // make Scan operator available in this test.
  test.AddCustomOpRegistry(op_registry);

  test.AddAttribute("body", proto);
  test.AddAttribute<std::vector<int64_t>>("scan_axes", {0});

  std::vector<int64_t> dims{1, 3};
  std::vector<float> X_data{1.f, 2.f, 3.f};

  test.AddInput<float>("initial_state_and_scan_inputs", dims, X_data);
  test.AddOutput<float>("final_state_and_scan_outputs", dims, X_data);

  test.Run();
}
}  // namespace Test
}  // namespace Lotus

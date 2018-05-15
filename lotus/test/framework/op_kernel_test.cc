#include "core/framework/inference_session.h"

#include "core/framework/function_kernel.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "gtest/gtest.h"

using namespace std;
using namespace Lotus::Logging;
using namespace LotusIR;

namespace Lotus {
namespace Test {

class XPUExecutionProvider : public IExecutionProvider {
 public:
  XPUExecutionProvider() = default;

  const LotusIR::GraphTransformer& GetTransformer() const override {
    return *graph_transformer_;
  }

  std::string Type() const override {
    return LotusIR::kCpuExecutionProvider;
  }

  Common::Status Compute(const LotusIR::Node& node, OpKernelContext* context) const override {
    UNUSED_PARAMETER(node);
    UNUSED_PARAMETER(context);
    return Status::OK();
  }

  Status CopyTensor(const Tensor& src, Tensor& dst) const override {
    UNUSED_PARAMETER(src);
    UNUSED_PARAMETER(dst);
    return Status::OK();
  }

  virtual const void* GetExecutionHandle() const noexcept override {
    // The XPU interface does not return anything interesting.
    return nullptr;
  }

 private:
  GraphTransformer* graph_transformer_ = nullptr;
};

TEST(OpKernelTest, CreateFunctionKernelTest) {
  LotusIR::Model model("test", true);
  auto graph = model.MainGraph();
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto input_arg = std::make_unique<NodeArg>("node_1_in_1", &tensor_int32);
  inputs.push_back(input_arg.get());
  auto output_arg = std::make_unique<NodeArg>("node_1_out_1", &tensor_int32);
  outputs.push_back(output_arg.get());
  auto node = graph->AddNode("node1", "op_not_exist", "this is going to call function kernel", inputs, outputs);
  node->SetExecutionProvider(LotusIR::kCpuExecutionProvider);
  AllocatorInfo alloc_info("CPU", AllocatorType::kArenaAllocator);
  CPUExecutionProviderInfo epi;
  CPUExecutionProvider exec_provider(epi);
  std::unique_ptr<OpKernel> kernel;
  auto status = KernelRegistry::Instance().CreateKernel(*node, &exec_provider, &kernel);
  ASSERT_TRUE(status.IsOK());
  ASSERT_EQ(typeid(FunctionKernel).name(), typeid(*kernel).name());

  node->SetExecutionProvider("XPUExecutionProvider");
  AllocatorInfo alloc_info_2("XPU", AllocatorType::kArenaAllocator);
  XPUExecutionProvider exec_provider_2;
  std::unique_ptr<OpKernel> kernel_2;
  auto status_2 = KernelRegistry::Instance().CreateKernel(*node, &exec_provider_2, &kernel_2);
  ASSERT_EQ(typeid(FunctionKernel).name(), typeid(*kernel_2).name());
  ASSERT_TRUE(status_2.IsOK());
  OpKernelContext* op_kernel_context = nullptr;
  auto status_3 = kernel_2->Compute(op_kernel_context);
  ASSERT_TRUE(status_3.IsOK());

  // TODO: add more cases for non-cpu execution providers.
}

}  // namespace Test
}  // namespace Lotus

#ifdef _MSC_VER
#pragma warning(push)
// 'identifier' : unreferenced formal parameter
#pragma warning(disable : 4100)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable : 4800)
#endif
#include "google/protobuf/util/message_differencer.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <iostream>
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "gtest/gtest.h"

namespace Lotus {
namespace Test {
class TestOpKernel : public OpKernel {
 public:
  TestOpKernel(const OpKernelInfo& p) : OpKernel(p) {}
  Status Compute(OpKernelContext* context) const {
    UNUSED_PARAMETER(context);
    return Status::OK();
  }
  Status ComputeAsync(OpKernelContext* context, DoneCallback done) const {
    UNUSED_PARAMETER(context);
    return Status::OK();
  }
};

TEST(SessionStateTest, AddGetKernelTest) {
  using google::protobuf::util::MessageDifferencer;

  SessionState s{10};  // dummy

  LotusIR::Model model("graph_1");
  auto graph = model.MainGraph();
  std::vector<LotusIR::NodeArg*> inputs;
  std::vector<LotusIR::NodeArg*> outputs;
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  LotusIR::NodeArg output_arg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  LotusIR::Node* p_node = graph->AddNode("node_1", "Variable", "node 1.", inputs, outputs);

  KernelDef kernel_def;
  OpKernelInfo p_info(*p_node, kernel_def, nullptr);
  unique_ptr<TestOpKernel> p_kernel;
  p_kernel.reset(new TestOpKernel(p_info));
  size_t orig_num_outputs = p_kernel->Node().OutputDefs().size();
  std::cout << "node_idx: " << p_node->Index() << std::endl;

  s.SetGraph(graph);
  s.AddKernel(p_node->Index(), std::move(p_kernel));
  auto test_kernel = s.GetKernel(p_node->Index());
  std::cout << "orig: " << orig_num_outputs << " new: " << test_kernel->Node().OutputDefs().size() << std::endl;
  EXPECT_EQ(orig_num_outputs, test_kernel->Node().OutputDefs().size());
}
}  // namespace Test
}  // namespace Lotus

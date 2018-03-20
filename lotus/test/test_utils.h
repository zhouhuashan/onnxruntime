#ifndef TEST_TEST_UTILS_H
#define TEST_TEST_UTILS_H

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"

namespace Lotus {
namespace Test {

class TestUtils {
  typedef std::shared_ptr<ExecutionFrame> ExecutionFramePtr;

 public:
  static ExecutionFramePtr CreateSingleNodeCPUExecutionFrame(
      const SessionState& session_state,
      std::unordered_map<std::string, MLValue> feeds,
      const std::vector<std::string> output_names) {
    return std::make_shared<ExecutionFrame>(
        feeds,
        output_names,
        session_state);
  }

  template <typename T>
  static Status PrepareTensor(const int index,
                              ExecutionFramePtr frame,
                              const std::vector<int64_t>& dims,
                              const std::vector<T>* value) {
    auto status = frame->AllocateTensorWithSelfOwnBuffer(
        index,
        DataTypeImpl::GetType<T>(),
        AllocatorManager::Instance()->GetArena(CPU).Info(),
        TensorShape(dims));
    if (!status.IsOK())
      return status;
    if (value) {
      auto tensor = frame->get_mutable_value<Tensor>(index);
      T* buffer = tensor->mutable_data<T>();
      for (int i = 0; i < value->size(); i++)
        buffer[i] = (*value)[i];
    }
    return Status::OK();
  }

  template <typename T>
  static Status PrepareIthInput(const LotusIR::Node& node,
                                const int i,
                                ExecutionFramePtr frame,
                                const std::vector<int64_t>& dims,
                                const std::vector<T>* value = nullptr) {
    LOTUS_ENFORCE(i >= 0 && i < node.InputDefs().size());
    return PrepareTensor(i, frame, dims, value);
  }

  template <typename T>
  static Status PrepareIthOutput(const LotusIR::Node& node,
                                 const int i,
                                 ExecutionFramePtr frame,
                                 const std::vector<int64_t>& dims,
                                 const std::vector<T>* value = nullptr) {
    LOTUS_ENFORCE(i >= 0 && i < node.OutputDefs().size());
    return PrepareTensor(i + (int)node.InputDefs().size(), frame, dims, value);
  }
};

#define CREATE_NODE(op_name, inputs, outputs)                 \
  LotusIR::Model model("test");                               \
  LotusIR::Graph* graph = model.MainGraph();                  \
  graph->AddNode("node1", op_name, op_name, inputs, outputs); \
  LotusIR::Node* node = graph->GetNode(graph->NumberOfNodes() - 1);
}  // namespace Test
}  // namespace Lotus

#endif  // !CORE_TEST_TEST_UTIL_H

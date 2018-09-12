#include "core/session/inference_session.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "gtest/gtest.h"
#include "test_utils.h"
using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::Logging;

namespace onnxruntime {
namespace Test {

class XPUExecutionProvider : public IExecutionProvider {
 public:
  XPUExecutionProvider() = default;

  std::string Type() const override {
    return onnxruntime::kCpuExecutionProvider;
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
};

}  // namespace Test
}  // namespace onnxruntime

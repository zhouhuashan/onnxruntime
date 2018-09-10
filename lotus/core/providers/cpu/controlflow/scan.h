#pragma once
#include <functional>
#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class SessionState;

class Scan final : public OpKernel {
 public:
  Scan(const OpKernelInfo& info) : OpKernel(info) {
    // make sure the attribute was present even though we don't need it here.
    // The GraphProto is processed by InferenceSession in order to setup a SessionState instance
    // with the necessary infrastructure to execute the subgraph.
    // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
    onnx::GraphProto proto;
    LOTUS_ENFORCE(info.GetAttr<onnx::GraphProto>("body", &proto).IsOK());
    (void)proto;
  }

  Status Compute(OpKernelContext* ctx) const override;

  static onnx::OpSchema GetScanOpSchema();

 private:
  Status ComputeImpl() const;
};
}  // namespace onnxruntime

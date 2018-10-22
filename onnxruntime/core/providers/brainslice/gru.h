#pragma once
#include "core/providers/brainslice/brainslice_kernel.h"

namespace onnxruntime {
namespace brainslice {
template <typename T>
class BrainSliceGRU : public BrainSliceOpKernel {
 public:
  explicit BrainSliceGRU(const OpKernelInfo& info);
  virtual Status Compute(OpKernelContext* context) const override;

 private:
  int64_t hidden_;
};
}  // namespace brainslice
}  // namespace onnxruntime

#pragma once
#include "cuda_pch.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"

namespace Lotus {

// Information needed to construct CUDA execution providers.
struct CUDAExecutionProviderInfo {
  std::string name;
  int device_id;
};

class CUDATransformer : public LotusIR::GraphTransformer {
 public:
  CUDATransformer(const std::string& name);
  Status Apply(LotusIR::Graph* graph, bool* modified) const override;
};

// Logical device representation.
class CUDAExecutionProvider : public IExecutionProvider {
 public:
  explicit CUDAExecutionProvider(const CUDAExecutionProviderInfo& info);

  const LotusIR::GraphTransformer& GetTransformer() const override {
    return transformer_;
  }

  AllocatorPtr GetAllocator() override {
    return arena_;
  }

  Status Compute(const LotusIR::Node& node, OpKernelContext* /*context*/) const override {
    return Common::Status(
        LOTUS, FAIL,
        "CUDA execution provider: can not run an op of type `" + node.OpType() + "'.");
  }

  std::string Type() const override {
    return LotusIR::kCudaExecutionProvider;
  }

  Status CopyTensor(const Tensor& src, Tensor& dst) override;

  virtual const void* GetExecutionHandle() const noexcept override {
    // The CUDA interface does not return anything interesting.
    return nullptr;
  }

  cublasHandle_t CublasHandle() const {
    return cublas_handle_;
  }

 private:
  CUDATransformer transformer_;
  int device_id_;
  cublasHandle_t cublas_handle_;
  ArenaPtr arena_;
};

}  // namespace Lotus

#pragma once
#include "cuda_pch.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "shared_inc/cuda_utils.h"

namespace Lotus {

// Information needed to construct CUDA execution providers.
struct CUDAExecutionProviderInfo {
  std::string name;
  int device_id = 0;
};

enum CUDAStreamType : int {
  kCudaStreamDefault = 0,
  kCudaStreamCopyIn,
  kCudaStreamCopyOut,
  kTotalCudaStreams,
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
  virtual ~CUDAExecutionProvider();

  const LotusIR::GraphTransformer& GetTransformer() const override {
    return transformer_;
  }

  Status Compute(const LotusIR::Node& node, OpKernelContext* /*context*/) const override {
    return Common::Status(
        LOTUS, FAIL,
        "CUDA execution provider: can not run an op of type `" + node.OpType() + "'.");
  }

  std::string Type() const override {
    return LotusIR::kCudaExecutionProvider;
  }

  Common::Status Sync() override;

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  virtual const void* GetExecutionHandle() const noexcept override {
    // The CUDA interface does not return anything interesting.
    return nullptr;
  }

  cublasHandle_t CublasHandle() const {
    return cublas_handle_;
  }

  cudaStream_t GetStream(int queue_id) const {
    LOTUS_ENFORCE(queue_id >= 0 && queue_id < kTotalCudaStreams);
    return streams_[queue_id];
  }

  const float* GetConstOnes(size_t count) {
    if (!constant_ones_)
      constant_ones_ = Cuda::CreateConstantOnesF();

    return constant_ones_->GetBuffer(count);
  }

 private:
  CUDATransformer transformer_;
  int device_id_;
  cublasHandle_t cublas_handle_;
  cudaStream_t streams_[kTotalCudaStreams];
  std::unique_ptr<Cuda::IConstantBuffer<float>> constant_ones_;
};

}  // namespace Lotus

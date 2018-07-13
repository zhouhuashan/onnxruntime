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

  Status Sync() override;

  Status OnRunStart() override;

  Status OnRunEnd() override;

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  virtual const void* GetExecutionHandle() const noexcept override {
    // The CUDA interface does not return anything interesting.
    return nullptr;
  }

  cublasHandle_t CublasHandle() const {
    return cublas_handle_;
  }

  cudnnHandle_t CudnnHandle() const {
    return cudnn_handle_;
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

  void AddDeferredReleaseCPUPtr(void* p) {
    // when not running in InferenceSession (e.g. Test)
    // it's OK to not remember the deferred release ptr
    // as the actual memory will be cleaned in arena allocator dtor
    if (current_deferred_release_event)
      deferred_release_cpu_ptr[current_deferred_release_event].push_back(p);
  }

 private:
  CUDATransformer transformer_;
  int device_id_;
  cublasHandle_t cublas_handle_ = nullptr;
  cudnnHandle_t cudnn_handle_ = nullptr;
  cudaStream_t streams_[kTotalCudaStreams];
  std::unique_ptr<Cuda::IConstantBuffer<float>> constant_ones_;

  // deferred release for temporary CPU pinned memory used in cudaMemcpyAsync
  // note that cudaEvent will be assigned at OnRunEnd()
  cudaEvent_t current_deferred_release_event = nullptr;
  std::unordered_map<cudaEvent_t, std::vector<void*>> deferred_release_cpu_ptr;
};

}  // namespace Lotus

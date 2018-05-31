#pragma once
#include "core/framework/tensor.h"
#include "cuda_execution_provider.h"
namespace Lotus {

class CUDAFence : public IFence {
 public:
  CUDAFence(const CUDAExecutionProvider* provider);
  virtual ~CUDAFence();
  virtual void BeforeUsingAsInput(LotusIR::ProviderType provider_type, int queue_id) override;
  virtual void BeforeUsingAsOutput(LotusIR::ProviderType provider_type, int queue_id) override;
  virtual void AfterUsedAsInput(int queue_id) override;
  virtual void AfterUsedAsOutput(int queue_id) override;

 private:
  cudaEvent_t read_event_;
  cudaEvent_t write_event_;
  const CUDAExecutionProvider* provider_;
};

}  // namespace Lotus

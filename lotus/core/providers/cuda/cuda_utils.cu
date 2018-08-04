// Thrust code needs to be compiled with nvcc
#include <memory>
#include <mutex>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace Lotus {
namespace Cuda {

template <typename T>
class ConstantBufferImpl : public IConstantBuffer<T> {
 public:
  ConstantBufferImpl(T val) : val_(val) {}

  virtual const T* GetBuffer(size_t count) {
    while (buffer_.size() < count) {
      std::lock_guard<std::mutex> lock(mutex_);
      buffer_.resize(count);
      thrust::fill(buffer_.begin(), buffer_.end(), val_);
    }
    return buffer_.data().get();
  }

 private:
  thrust::device_vector<T> buffer_;
  T val_;
  std::mutex mutex_;
};

std::unique_ptr<IConstantBuffer<float>> CreateConstantOnesF() {
  return std::make_unique<ConstantBufferImpl<float>>(1.0f);
}

}  // namespace Cuda
}  // namespace Lotus

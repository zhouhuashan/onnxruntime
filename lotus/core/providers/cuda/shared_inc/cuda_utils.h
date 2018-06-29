// for things shared between nvcc and Lotus
// as currently nvcc cannot compile all Lotus headers

#pragma once
#include <memory>
#include <vector>
#include "fast_divmod.h"

namespace Lotus {
namespace Cuda {

enum class SimpleBroadcast : size_t {
  NoBroadcast = (size_t)-1,
  LeftScalar = (size_t)-2,
  RightScalar = (size_t)-3,
};

template <typename T>
class IConstantBuffer {
 public:
  virtual ~IConstantBuffer(){};
  virtual const T* GetBuffer(size_t count) = 0;
};

std::unique_ptr<IConstantBuffer<float>> CreateConstantOnesF();

class FastDivModStrides {
 public:
  FastDivModStrides(std::vector<int64_t> dims, size_t rank = 0) {
    int stride = 1;
    if (dims.size() > rank) rank = dims.size();
    strides_.resize(rank);
    for (int i = 0; i < rank; i++) {
      strides_[rank - 1 - i] = fast_divmod(stride);
      if (i < dims.size() - 1) {
        stride *= static_cast<int>(dims[dims.size() - 1 - i]);
      }
    }
  }

  const std::vector<fast_divmod>& GetStrides() const { return strides_; }

 private:
  std::vector<fast_divmod> strides_;
};

}  // namespace Cuda
}  // namespace Lotus

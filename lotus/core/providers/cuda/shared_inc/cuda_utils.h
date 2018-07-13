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

}  // namespace Cuda
}  // namespace Lotus

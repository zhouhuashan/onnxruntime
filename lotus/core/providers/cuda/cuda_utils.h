#pragma once
#include <memory>
namespace Lotus {
namespace Cuda {

template <typename T>
class IConstantBuffer {
 public:
  virtual ~IConstantBuffer(){};
  virtual const T* GetBuffer(size_t count) = 0;
};

IConstantBuffer<float>* CreateConstantOnesF();
}  // namespace Cuda
}  // namespace Lotus

#ifndef CORE_FRAMEWORK_ML_VALUE_H
#define CORE_FRAMEWORK_ML_VALUE_H

#include <string>
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor.h"

namespace Lotus {
class MLValue {
 public:
  MLValue() : pData_(nullptr), type_(nullptr) {}
  virtual ~MLValue() { Reset(); }

  void Init(void* pData, MLDataType type, DeleteFunc deleter) {
    pData_.reset(pData, deleter);
    type_ = type;
  }

  void Reset() {
    pData_ = nullptr;
    type_ = nullptr;
  }

  bool IsAllocated() {
    return pData_ && type_;
  }

  template <typename T>
  const T& Get() const {
    LOTUS_ENFORCE(DataTypeImpl::GetType<T>() == type_);
    return *static_cast<T*>(pData_.get());
  }

  template <typename T>
  T* GetMutable() {
    LOTUS_ENFORCE(DataTypeImpl::GetType<T>() == type_);
    return static_cast<T*>(pData_.get());
  }

  bool IsTensor() {
    return DataTypeImpl::GetType<Tensor>() == type_;
  }

 private:
  std::shared_ptr<void> pData_;
  MLDataType type_;
};
}  // namespace Lotus

#endif  // CORE_FRAMEWORK_ML_VALUE_H

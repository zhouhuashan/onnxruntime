// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
/**
   Represents both tensors and non-tensors.
*/
class MLValue {
 private:
  template <typename Result, typename TReg>
  struct Fetcher {
    static const Result& Get(const MLValue& ml_value) {
      ONNXRUNTIME_ENFORCE(DataTypeImpl::GetType<TReg>() == ml_value.type_,
                  DataTypeImpl::GetType<TReg>(), " != ", ml_value.type_);
      return *static_cast<Result*>(ml_value.data_.get());
    }
    static Result* GetMutable(MLValue& ml_value) {
      ONNXRUNTIME_ENFORCE(DataTypeImpl::GetType<TReg>() == ml_value.type_,
                  DataTypeImpl::GetType<TReg>(), " != ", ml_value.type_);
      return static_cast<Result*>(ml_value.data_.get());
    }
  };

  template <typename T, typename... Types>
  struct TypeRegistrationDispatcher;

  template <typename T>
  struct TypeRegistrationDispatcher<T> : public Fetcher<T, T> {
  };

  template <typename T, typename... Types>
  struct TypeRegistrationDispatcher<TypeRegister<T, Types...>> : public Fetcher<T, TypeRegister<T, Types...>> {
  };

  template <typename T, const char D[], const char N[], typename... Params>
  struct TypeRegistrationDispatcher<OpaqueRegister<T, D, N, Params...>> : public Fetcher<T, OpaqueRegister<T, D, N, Params...>> {
  };

 public:
  MLValue() : data_(nullptr) {}
  virtual ~MLValue() = default;

  MLValue(void* pData, MLDataType type, DeleteFunc deleter) {
    Init(pData, type, deleter);
  }

  void Init(void* pData, MLDataType type, DeleteFunc deleter) {
    data_.reset(pData, deleter);
    type_ = type;
  }

  bool IsAllocated() const {
    return data_ && type_;
  }

  template <typename T>
  const auto& Get() const {
    return TypeRegistrationDispatcher<T>::Get(*this);
  }

  template <typename T>
  auto* GetMutable() {
    return TypeRegistrationDispatcher<T>::GetMutable(*this);
  }

  bool IsTensor() const {
    return DataTypeImpl::GetType<Tensor>() == type_;
  }

  MLDataType Type() const {
    return type_;
  }

  Fence_t Fence() const {
    return fence_.get();
  }

  void SetFence(FencePtr fence) {
    fence_ = fence;
  }

  void ShareFenceWith(MLValue& v) {
    fence_ = v.fence_;
  }

 private:
  std::shared_ptr<void> data_;
  MLDataType type_{nullptr};
  FencePtr fence_;
};
}  // namespace onnxruntime

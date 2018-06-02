#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "gsl/span"

#include "core/framework/arena.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor_shape.h"
using namespace onnx;

namespace Lotus {

// TODO: Do we need this class or is IAllocator::MakeUniquePtr sufficient/better
class BufferDeleter {
 public:
  BufferDeleter() : alloc_(nullptr) {}
  BufferDeleter(AllocatorPtr alloc)
      : alloc_(alloc) {}

  void operator()(void* p) const {
    if (alloc_)
      alloc_->Free(p);
  }

 private:
  // TODO: we may need consider the lifetime of alloc carefully
  // The alloc_ here is the allocator that used to allocate the buffer
  // And need go with the unique_ptr together. If it is using our internal
  // allocator, it is ok as our allocators are global managed. But if it
  // is provide by user, user need to be very careful about it.
  // A weak_ptr may be a choice to reduce the impact, but that require to
  // change our current allocator mgr to use shared_ptr. Will revisit it
  // later.
  AllocatorPtr alloc_;
};

typedef std::unique_ptr<void, BufferDeleter> BufferUniquePtr;
using BufferNakedPtr = void*;

/*
We want to keep tensor as simple as possible, it is just a placeholder 
for a piece of memory, with additional shape information.
Memory is owned and managed by Executor / Workspace, so Tensor just uses 
it, and won't do any allocation / release.
*/
class Tensor {
  friend class TensorUtil;
  friend class MLValue;
  friend class ExecutionFrame;

 public:
  // Create tensor with given type, shape, pre-allocate memory and allocator info.
  Tensor(MLDataType p_type,
         const TensorShape& shape,
         BufferNakedPtr p_data,
         const AllocatorInfo& alloc,
         AllocatorPtr deleter = nullptr,
         int64_t offset = 0);

  virtual ~Tensor();

  // Copy constructor and assign op will just pass the shape and memory
  // reference to another tensor. Not deep clone/copy.
  Tensor(const Tensor& src);
  Tensor& ShallowCopy(const Tensor& other);

  Tensor(Tensor&& other);

  Tensor& operator=(Tensor&& other);

  // Returns the data type.
  MLDataType DataType() const { return dtype_; }

  // Returns the shape of the tensor.
  const TensorShape& Shape() const noexcept { return shape_; }

  // Returns the location of the tensor's memory
  const AllocatorInfo& Location() const { return alloc_info_; }

  // May return nullptr if tensor size is zero
  template <typename T>
  T* MutableData() {
    // Type check
    LOTUS_ENFORCE(DataTypeImpl::GetType<T>() == dtype_, "Tensor type mismatch. ",
                  DataTypeImpl::GetType<T>(), "!=", dtype_);
    return reinterpret_cast<T*>(static_cast<char*>(GetRaw()) + byte_offset_);
  }

  // May return nullptr if tensor size is zero
  template <typename T>
  gsl::span<T> MutableDataAsSpan() {
    // Type check
    LOTUS_ENFORCE(DataTypeImpl::GetType<T>() == dtype_, "Tensor type mismatch. ",
                  DataTypeImpl::GetType<T>(), "!=", dtype_);
    T* data = reinterpret_cast<T*>(static_cast<char*>(GetRaw()) + byte_offset_);
    return gsl::make_span(data, shape_.Size());
  }

  template <typename T>
  const T* Data() const {
    // Type check
    LOTUS_ENFORCE(DataTypeImpl::GetType<T>() == dtype_, "Tensor type mismatch. ",
                  DataTypeImpl::GetType<T>(), "!=", dtype_);
    return reinterpret_cast<const T*>(static_cast<char*>(GetRaw()) + byte_offset_);
  }

  template <typename T>
  gsl::span<const T> DataAsSpan() const {
    // Type check
    LOTUS_ENFORCE(DataTypeImpl::GetType<T>() == dtype_, "Tensor type mismatch. ",
                  DataTypeImpl::GetType<T>(), "!=", dtype_);
    const T* data = reinterpret_cast<const T*>(static_cast<char*>(GetRaw()) + byte_offset_);
    return gsl::make_span(data, shape_.Size());
  }

  void* MutableDataRaw(MLDataType type) {
    LOTUS_ENFORCE(type == dtype_, "Tensor type mismatch.", type, "!=", dtype_);
    return GetRaw();
  }

  const void* DataRaw(MLDataType type) const {
    LOTUS_ENFORCE(type == dtype_, "Tensor type mismatch.", type, "!=", dtype_);
    return GetRaw();
  }

  void* MutableDataRaw() noexcept {
    return GetRaw();
  }

  const void* DataRaw() const noexcept {
    return GetRaw();
  }

  /**
  * Resizes the tensor without touching underlying storage.
  * This requires the total size of the tensor to remains constant.
  * @warning this function is NOT thread-safe.
  */
  inline void Reshape(const TensorShape& new_shape) {
    LOTUS_ENFORCE(shape_.Size() == new_shape.Size(),
                  "Tensor size (" + std::to_string(shape_.Size()) +
                      ") != new size (" + std::to_string(new_shape.Size()) + ")");
    shape_ = new_shape;
  }

  const size_t Size() const noexcept {
    return shape_.Size() * dtype_->Size();
  }

  // More API methods.
 private:
  void Init(MLDataType p_type,
            const TensorShape& shape,
            void* p_raw_data,
            const AllocatorInfo& alloc,
            AllocatorPtr deleter,
            int64_t offset = 0);

  void* GetRaw() const noexcept {
    return p_data_;
  }

  void* p_data_;
  // if buffer_deleter_ is null, it means tensor does not own the buffer.
  // otherwise tensor will use the deleter to release the buffer when
  // tensor is released.
  AllocatorPtr buffer_deleter_;

  TensorShape shape_;
  MLDataType dtype_;
  AllocatorInfo alloc_info_;
  int64_t byte_offset_;
};

}  // namespace Lotus

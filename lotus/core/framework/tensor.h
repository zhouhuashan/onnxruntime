#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "core/framework/allocator.h"
#include "core/framework/data_types.h"

using namespace onnx;

namespace Lotus {

// TODO(Task:130) Move TensorShape to separate.h / .cc
// TODO - Use a custom STL allocator to avoid heap allocations in the common case.
// We use negative numbers for unknown symbolic dimension. Each negative
// number represents a unique symbolic dimension.
// Private inheritance is used to prevent ambiguity of element versus dimension size
class TensorShape : private std::vector<int64_t> {
 public:
  TensorShape();

  TensorShape(const int64_t* dimension_sizes, size_t dimension_count);

  TensorShape(const std::vector<int64_t>& dims);

  TensorShape(const TensorShape& other);

  TensorShape(const std::vector<int64_t>& dims, size_t start, size_t end);

  /**
  Return the dimension specified by <idx>.
  */
  const int64_t& operator[](size_t idx) const {
    return std::vector<int64_t>::operator[](static_cast<int>(idx));
  }

  int64_t& operator[](size_t idx) {
    return std::vector<int64_t>::operator[](static_cast<int>(idx));
  }

  bool operator==(const TensorShape& other) const noexcept {
    auto thisVector = static_cast<const std::vector<int64_t>*>(this);
    auto otherVector = static_cast<const std::vector<int64_t>*>(&other);
    return *thisVector == *otherVector;
  }

  bool operator!=(const TensorShape& other) const noexcept {
    return !(*this == other);
  }

  const size_t NumDimensions() const {
    return size();
  }

  /**
  Copy dims into an array with given size
  */
  void CopyDims(int64_t* dims, size_t num_dims) const {
    memcpy(dims, data(), sizeof(value_type) * std::min(num_dims, NumDimensions()));
  }

  /**
  Return underlying vector representation.
  */
  const std::vector<int64_t>& GetDims() const { return *this; }

  /** 
  Return the total number of elements. Returns 1 for an empty (rank 0) TensorShape.
  */
  int64_t Size() const;

  /** 
  Return the total number of elements up to the specified dimension.
  @param dimension Return size up to this dimension. Value must be >= 0 and < this.Size().
  */
  int64_t SizeToDimension(size_t dimension) const;

  /**
  Return the total number of elements from the specified dimension to the end of the tensor shape.
  @param dimension Return size up to this dimension. 0 <= dimension < this.Size().
  */
  int64_t SizeFromDimension(size_t dimension) const;

  // Return a new TensorShape of the dimensions from dimstart to dimend.
  TensorShape Slice(size_t dimstart, size_t dimend) const;

  // Return a new TensorShape of the dimensions from dimstart to end.
  TensorShape Slice(size_t dimstart) const;

  // output dimensions nicely formatted
  std::string ToString() const;

  // Calculate size between start and end.
  // Assumes start and end are between 0 and dimensions.size(), inclusive, and that
  // start < end.
  int64_t SizeHelper(size_t start, size_t end) const;

  static const TensorShape& ReinterpretBaseType(const std::vector<int64_t>& dimensions) {
    static_assert(sizeof(TensorShape) == sizeof(std::vector<int64_t>), "Size of TensorShape prevents safe casting from vector");
    return *static_cast<const TensorShape*>(&dimensions);
  }
};

// operator<< to nicely output to a stream
std::ostream& operator<<(std::ostream& out, const TensorShape& shape);

class BufferDeleter {
 public:
  BufferDeleter() : alloc_(nullptr) {}
  BufferDeleter(IAllocator* alloc)
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
  IAllocator* alloc_;
};

typedef std::unique_ptr<void, BufferDeleter> BufferUniquePtr;
typedef void* BufferNakedPtr;

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
  // Create an empty tensor with float type.
  // empty tensor is a tensor with 1-d shape (0,), and 0 elements.
  Tensor();
  // Create a empty tensor with given type
  Tensor(MLDataType p_type);
  // Create tensor with given type, shape, pre-allocate memory and allocator info.
  Tensor(MLDataType p_type,
         const TensorShape& shape,
         BufferNakedPtr p_data,
         const AllocatorInfo& alloc,
         IAllocator* deleter = nullptr,
         const int64_t offset = 0);

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
  const TensorShape& Shape() const { return shape_; }

  // Returns the location of the tensor's memory
  const AllocatorInfo& Location() const { return alloc_info_; }

  template <typename T>
  T* MutableData() {
    // Type check
    LOTUS_ENFORCE(DataTypeImpl::GetType<T>() == dtype_, "Tensor type mismatch. ",
                  DataTypeImpl::GetType<T>(), "!=", dtype_);
    return reinterpret_cast<T*>(static_cast<char*>(GetRaw()) + byte_offset_);
  }

  template <typename T>
  const T* Data() const {
    // Type check
    LOTUS_ENFORCE(DataTypeImpl::GetType<T>() == dtype_, "Tensor type mismatch. ",
                  DataTypeImpl::GetType<T>(), "!=", dtype_);
    return reinterpret_cast<const T*>(static_cast<char*>(GetRaw()) + byte_offset_);
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

  // More API methods.
 private:
  void Init(MLDataType p_type,
            const TensorShape& shape,
            void* p_raw_data,
            const AllocatorInfo& alloc,
            IAllocator* deleter,
            const int64_t offset = 0);

  void* GetRaw() const noexcept {
    return p_data_;
  }

  void* p_data_;
  // if buffer_deleter_ is null, it means tensor does not own the buffer.
  // otherwise tensor will use the deleter to release the buffer when
  // tensor is released.
  IAllocator* buffer_deleter_;

  TensorShape shape_;
  MLDataType dtype_;
  AllocatorInfo alloc_info_;
  int64_t byte_offset_;
};

}  // namespace Lotus

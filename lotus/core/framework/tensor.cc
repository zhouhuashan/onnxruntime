#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"

namespace Lotus {

size_t TensorShape::SizeHelper(const std::vector<int64_t>& dimensions, size_t start, size_t end) {
  size_t size = 1;
  for (size_t i = start; i < end; i++) {
    LOTUS_ENFORCE(dimensions[i] >= 0, "Can't calculate size for a un-resolved tensor shape");
    size *= dimensions[i];
  }
  return size;
}

TensorShape::TensorShape() : TensorShape(std::vector<int64_t>()) {
}

TensorShape::TensorShape(const std::vector<int64_t>& dims) : dims_(dims) {
}

TensorShape::TensorShape(const TensorShape& other) {
  dims_.assign(other.dims_.begin(), other.dims_.end());
}

const int64_t TensorShape::operator[](int idx) const {
  //Since we don't have status in return value,
  //the caller should be responsible for invalid idx.
  //In that case, stl throws an exception.
  return dims_.at(idx);
}

size_t TensorShape::Size() const {
  size_t size = SizeHelper(dims_, 0, dims_.size());
  //should we cache the size? as multiple operation may be expensive.
  return size;
}

size_t TensorShape::SizeToDimension(size_t dimension) const {
  const size_t num_dims = dims_.size();
  LOTUS_ENFORCE(dimension <= num_dims,
                "Invalid dimension of %d for SizeToDimension. Tensor has %d dimensions.", dimension, num_dims);

  size_t size = SizeHelper(dims_, 0, dimension);
  return size;
}

size_t TensorShape::SizeFromDimension(size_t dimension) const {
  const size_t num_dims = dims_.size();
  LOTUS_ENFORCE(dimension < num_dims,
                "Invalid dimension of %d for SizeFromDimension. Tensor has %d dimensions.", dimension, num_dims);

  size_t size = SizeHelper(dims_, dimension, num_dims);
  return size;
}

TensorShape TensorShape::Slice(size_t dimstart, size_t dimend) const {
  LOTUS_ENFORCE(dimstart >= 0 && dimstart <= dimend && dimend <= dims_.size(), "Invalid tensor shape slice argument.");
  return TensorShape(std::vector<int64_t>(dims_.begin() + dimstart, dims_.begin() + dimend));
}

TensorShape TensorShape::Slice(size_t dimstart) const {
  return Slice(dimstart, dims_.size());
}

// output dimensions
std::string TensorShape::ToString() const {
  std::string result;
  result.reserve(2 + dims_.size() * 5);  // generous calculation '{' + '}' and 4 digits + ',' for each entry

  result.append("{");

  bool first = true;
  for (auto dim : dims_) {
    if (!first) {
      result.append(",");
    }

    result.append(std::to_string(dim));

    first = false;
  }

  result.append("}");

  return result;
}

// operator<< to nicely output to a stream
std::ostream& operator<<(std::ostream& out, const TensorShape& shape) {
  return (out << shape.ToString());
}

Tensor::Tensor() : alloc_info_(AllocatorManager::Instance().GetArena(CPU).Info()),
                   p_unique_data_(BufferUniquePtr(nullptr, BufferDeleter())) {
  Init(DataTypeImpl::GetType<float>(),
       TensorShape(std::vector<int64_t>(1, 0)),
       UNKNOWN,
       nullptr,
       AllocatorManager::Instance().GetArena(CPU).Info(),
       0);
}

Tensor::Tensor(MLDataType p_type) : alloc_info_(AllocatorManager::Instance().GetArena(CPU).Info()),
                                    p_unique_data_(BufferUniquePtr(nullptr, BufferDeleter())) {
  Init(p_type,
       TensorShape(std::vector<int64_t>(1, 0)),
       UNKNOWN,
       nullptr,
       AllocatorManager::Instance().GetArena(CPU).Info(),
       0);
}

Tensor::Tensor(MLDataType p_type,
               const TensorShape& shape,
               BufferNakedPtr p_data,
               const AllocatorInfo& alloc,
               const int64_t offset)
    : alloc_info_(alloc),
      p_unique_data_(BufferUniquePtr(nullptr, BufferDeleter())) {
  Init(p_type,
       shape,
       PREALLOCATEDBUFFER,
       p_data,
       alloc,
       offset);
}

Tensor::Tensor(MLDataType p_type,
               const TensorShape& shape,
               BufferUniquePtr p_data,
               const AllocatorInfo& alloc,
               const int64_t offset)
    : alloc_info_(alloc),
      p_unique_data_(std::move(p_data)) {
  Init(p_type,
       shape,
       OWNEDBUFFER,
       nullptr,
       alloc,
       offset);
}

void Tensor::Init(MLDataType p_type,
                  const TensorShape& shape,
                  BufferStrategy strategy,
                  BufferNakedPtr p_raw_data,
                  const AllocatorInfo& alloc,
                  const int64_t offset) {
  dtype_ = p_type;
  shape_ = shape;
  buffer_strategy_ = strategy;
  p_naked_data_ = p_raw_data;
  alloc_info_ = alloc;
  byte_offset_ = offset;
}

Tensor::Tensor(Tensor&& other)
    : dtype_(other.dtype_),
      shape_(other.shape_),
      alloc_info_(other.alloc_info_),
      byte_offset_(other.byte_offset_),
      buffer_strategy_(other.buffer_strategy_) {
  if (other.buffer_strategy_ == OWNEDBUFFER) {
    p_unique_data_ = std::move(other.p_unique_data_);
    p_naked_data_ = nullptr;
  } else {
    p_naked_data_ = other.p_naked_data_;
    p_unique_data_ = nullptr;
  }

  other.dtype_ = DataTypeImpl::GetType<float>();
  other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
  other.buffer_strategy_ = UNKNOWN;
  other.byte_offset_ = 0;
  other.p_unique_data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) {
  if (this != &other) {
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    alloc_info_ = other.alloc_info_;
    byte_offset_ = other.byte_offset_;
    buffer_strategy_ = other.buffer_strategy_;
    if (other.buffer_strategy_ == OWNEDBUFFER) {
      p_unique_data_ = std::move(other.p_unique_data_);
      p_naked_data_ = nullptr;
    } else {
      p_naked_data_ = other.p_naked_data_;
      p_unique_data_ = nullptr;
    }

    other.dtype_ = DataTypeImpl::GetType<float>();
    other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
    other.buffer_strategy_ = UNKNOWN;
    other.byte_offset_ = 0;
    other.p_unique_data_ = nullptr;
  }
  return *this;
}

Tensor::Tensor(const Tensor& src)
    : dtype_(src.dtype_), alloc_info_(src.alloc_info_), shape_(src.shape_), byte_offset_(src.byte_offset_) {
  // it may be better to refactor it a little bit to make it a compile error
  // but right now just keep it simple first.
  LOTUS_ENFORCE(src.buffer_strategy_ != OWNEDBUFFER,
                "Can't copy tensor with its owned buffer. Please transfer ownership by move");

  if (src.buffer_strategy_ == PREALLOCATEDBUFFER) {
    buffer_strategy_ = PREALLOCATEDBUFFER;
    p_naked_data_ = src.p_naked_data_;
  } else {
    buffer_strategy_ = UNKNOWN;
    p_naked_data_ = nullptr;
    p_unique_data_ = nullptr;
  }
}

Tensor& Tensor::ShallowCopy(const Tensor& other) {
  // similar as above
  LOTUS_ENFORCE(other.buffer_strategy_ != OWNEDBUFFER,
                "Can't copy tensor with its owned buffer. Please transfer ownership by move");

  if (this != &other) {
    dtype_ = other.dtype_;
    alloc_info_ = other.alloc_info_;
    shape_ = other.shape_;
    byte_offset_ = other.byte_offset_;
    buffer_strategy_ = other.buffer_strategy_;
    p_naked_data_ = other.p_naked_data_;
    p_naked_data_ = nullptr;
  }
  return *this;
}

}  // namespace Lotus

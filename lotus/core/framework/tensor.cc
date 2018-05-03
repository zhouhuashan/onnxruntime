#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"

namespace Lotus {

TensorShape::TensorShape() {}

TensorShape::TensorShape(const std::vector<int64_t>& dims) : std::vector<int64_t>(dims) {
}

TensorShape::TensorShape(const int64_t* dimension_sizes, size_t dimension_count) : std::vector<int64_t>(dimension_count) {
  for (size_t i = 0; i < dimension_count; ++i) {
    (*this)[i] = dimension_sizes[i];
  }
}

TensorShape::TensorShape(const TensorShape& other) : std::vector<int64_t>(other) {
}

TensorShape::TensorShape(const std::vector<int64_t>& dims, size_t start, size_t end) {
  assign(dims.begin() + start, dims.begin() + end);
}

/**
Return the total number of elements. Returns 1 for an empty (rank 0) TensorShape.
*/
int64_t TensorShape::Size() const {
  size_t arraySize = size();
  int64_t size = SizeHelper(0, arraySize);
  //should we cache the size? as multiple operation may be expensive.
  return size;
}

int64_t TensorShape::SizeToDimension(size_t dimension) const {
  const size_t num_dims = size();
  LOTUS_ENFORCE(dimension <= num_dims,
                "Invalid dimension of ", dimension, " for SizeFromDimension. Tensor has ",
                num_dims, " dimensions.");

  int64_t size = SizeHelper(0, dimension);
  return size;
}

int64_t TensorShape::SizeFromDimension(size_t dimension) const {
  const size_t num_dims = size();
  LOTUS_ENFORCE(dimension <= num_dims,
                "Invalid dimension of ", dimension, " for SizeFromDimension. Tensor has ",
                num_dims, " dimensions.");

  int64_t size = SizeHelper(dimension, num_dims);
  return size;
}

TensorShape TensorShape::Slice(size_t dimstart, size_t dimend) const {
  LOTUS_ENFORCE(dimstart >= 0 && dimstart <= dimend && dimend <= size(),
                "Invalid tensor shape slice argument.");
  return TensorShape(*this, dimstart, dimend);
}

TensorShape TensorShape::Slice(size_t dimstart) const {
  return Slice(dimstart, size());
}

// output dimensions
std::string TensorShape::ToString() const {
  std::string result;

  result.append("{");
  bool first = true;
  for (auto dim : (*this)) {
    if (!first) {
      result.append(",");
    }

    result.append(std::to_string(dim));
    first = false;
  }
  result.append("}");

  return result;
}

int64_t TensorShape::SizeHelper(size_t start, size_t end) const {
  // Must return 1 for an empty sequence
  int64_t size = 1;
  for (size_t i = start; i < end; i++) {
    if ((*this)[i] < 0) return -1;
    size *= (*this)[i];
  }
  return size;
}

// operator<< to nicely output to a stream
std::ostream& operator<<(std::ostream& out, const TensorShape& shape) {
  return (out << shape.ToString());
}



Tensor::Tensor(MLDataType p_type,
               const TensorShape& shape,
               BufferNakedPtr p_data,
               const AllocatorInfo& alloc,
               IAllocator* deleter,
               const int64_t offset)
    : alloc_info_(alloc) {
  Init(p_type, shape, p_data, alloc, deleter, offset);
}

void Tensor::Init(MLDataType p_type,
                  const TensorShape& shape,
                  void* p_raw_data,
                  const AllocatorInfo& alloc,
                  IAllocator* deleter,
                  const int64_t offset) {
  dtype_ = p_type;
  shape_ = shape;
  p_data_ = p_raw_data;
  // if caller passed in a deleter, that means this tensor own this buffer
  // we will release the buffer when this tensor is deconstructed.
  buffer_deleter_ = deleter;
  // for string tensors, if this tensor own the buffer (caller passed in the deleter)
  // do the placement new for strings on pre-allocated buffer.
  if (buffer_deleter_ && dtype_ == DataTypeImpl::GetType<std::string>()) {
    std::string* ptr = static_cast<std::string*>(p_data_);
    for (int64_t i = 0, n = shape.Size(); i < n; ++i) {
      new (ptr + i) std::string();
    }
  }
  alloc_info_ = alloc;
  byte_offset_ = offset;
}

Tensor::Tensor(Tensor&& other)
    : dtype_(other.dtype_),
      shape_(other.shape_),
      alloc_info_(other.alloc_info_),
      byte_offset_(other.byte_offset_),
      p_data_(other.p_data_),
      buffer_deleter_(other.buffer_deleter_) {
  other.dtype_ = DataTypeImpl::GetType<float>();
  other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
  other.p_data_ = nullptr;
  other.buffer_deleter_ = nullptr;
  other.byte_offset_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) {
  if (this != &other) {
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    alloc_info_ = other.alloc_info_;
    byte_offset_ = other.byte_offset_;
    p_data_ = other.p_data_;
    buffer_deleter_ = other.buffer_deleter_;

    other.dtype_ = DataTypeImpl::GetType<float>();
    other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
    other.p_data_ = nullptr;
    other.byte_offset_ = 0;
    other.buffer_deleter_ = nullptr;
  }
  return *this;
}

Tensor::Tensor(const Tensor& src)
    : dtype_(src.dtype_), alloc_info_(src.alloc_info_), shape_(src.shape_), byte_offset_(src.byte_offset_) {
  // it may be better to refactor it a little bit to make it a compile error
  // but right now just keep it simple first.
  LOTUS_ENFORCE(src.buffer_deleter_ == nullptr,
                "Can't copy tensor with its owned buffer. Please transfer ownership by move.");

  p_data_ = src.p_data_;
  buffer_deleter_ = nullptr;
}

Tensor::~Tensor() {
  if (buffer_deleter_) {
    // if current tensor is responsible for delete the buffer
    // and it is a string tensor, need to explict call string's
    // deconstructor.
    if (dtype_ == DataTypeImpl::GetType<std::string>()) {
      std::string* ptr = static_cast<std::string*>(p_data_);
      for (int i = 0; i < shape_.Size(); i++)
        ptr[i].~string();
    }
    buffer_deleter_->Free(p_data_);
  }
}

Tensor& Tensor::ShallowCopy(const Tensor& other) {
  // similar as above
  LOTUS_ENFORCE(other.buffer_deleter_ == nullptr,
                "Can't copy tensor with its owned buffer. Please transfer ownership by move.");

  if (this != &other) {
    dtype_ = other.dtype_;
    alloc_info_ = other.alloc_info_;
    shape_ = other.shape_;
    byte_offset_ = other.byte_offset_;
    p_data_ = other.p_data_;
    buffer_deleter_ = nullptr;
  }
  return *this;
}

}  // namespace Lotus

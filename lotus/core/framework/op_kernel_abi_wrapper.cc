//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#include "core/framework/op_kernel.h"
#include "core/framework/execution_frame.h"
#include "op_kernel_abi_wrapper.h"
#include "core/inc/op_kernel_author_helper.h"
#include <limits>
#include <assert.h>

namespace Lotus {

inline MLStatus ToABIStatus(StatusCode statusCode) {
  return static_cast<MLStatus>(statusCode);
}

inline MLStatus ToABIStatus(Status status) {
  return static_cast<MLStatus>(status.Code());
}

//
// Traits for numeric attribute types
//
template <MLAttributeType T>
struct MLAttributeTypeTraits {
};

template <>
struct MLAttributeTypeTraits<MLAttributeType::kFloat> {
  typedef float Type;
  static const AttributeProto_AttributeType ProtoType = AttributeProto_AttributeType_FLOAT;
};

template <>
struct MLAttributeTypeTraits<MLAttributeType::kInt> {
  typedef int64_t Type;
  static const AttributeProto_AttributeType ProtoType = AttributeProto_AttributeType_INT;
};

template <MLAttributeType T>
struct MLAttributeArrayTypeTraits {
};

template <>
struct MLAttributeArrayTypeTraits<MLAttributeType::kFloatArray> {
  typedef float Type;
  static const AttributeProto_AttributeType ProtoType = AttributeProto_AttributeType_FLOATS;
};

template <>
struct MLAttributeArrayTypeTraits<MLAttributeType::kIntArray> {
  typedef int64_t Type;
  static const AttributeProto_AttributeType ProtoType = AttributeProto_AttributeType_INTS;
};

inline AttributeProto_AttributeType ToProto(MLAttributeType type) {
  switch (type) {
    case MLAttributeType::kFloat:
      return MLAttributeTypeTraits<MLAttributeType::kFloat>::ProtoType;

    case MLAttributeType::kInt:
      return MLAttributeTypeTraits<MLAttributeType::kInt>::ProtoType;

    case MLAttributeType::kFloatArray:
      return MLAttributeArrayTypeTraits<MLAttributeType::kFloatArray>::ProtoType;

    case MLAttributeType::kIntArray:
      return MLAttributeArrayTypeTraits<MLAttributeType::kIntArray>::ProtoType;

    default:
      return AttributeProto_AttributeType_UNDEFINED;
  }
}

OpKernelInfoWrapper::OpKernelInfoWrapper(const OpKernelInfo* kernelInfo) {
  impl_ = kernelInfo;
}

ML_API_IMP(OpKernelInfoWrapper::GetAttributeElementCount)(
    MLAttributeType type,
    const char* name,
    uint32_t* element_count) const noexcept {

  *element_count = 0;

  try {
    *element_count = impl_->GetPrimitiveAttrElementCount(
        ToProto(type),
        std::string(name));

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

template <MLAttributeType T>
MLStatus OpKernelInfoWrapper::GetAttributeArrayHelper(
    const char* name,
    uint32_t element_count,
    uint32_t element_byte_size,
    void* values) const {
  typedef typename MLAttributeArrayTypeTraits<T>::Type elementType_t;
  ML_CHECK_BOOL(sizeof(elementType_t) == element_byte_size);

  ML_CHECK_STATUS(ToABIStatus(impl_->GetAttrs(name, gsl::span<elementType_t>(static_cast<typename MLAttributeArrayTypeTraits<T>::Type*>(values), element_count))));
  return MLStatus::OK;
}

ML_API_IMP(OpKernelInfoWrapper::GetAttribute)(
    const char* name,
    MLAttributeType type,
    uint32_t element_count,
    uint32_t element_byte_size,
    void* value) const noexcept {
  try {
    switch (type) {
      case MLAttributeType::kFloat:
        ML_CHECK_BOOL(element_count == 1);
        return GetAttributeHelper<MLAttributeType::kFloat>(name, element_byte_size, value);

      case MLAttributeType::kInt:
        ML_CHECK_BOOL(element_count == 1);
        return GetAttributeHelper<MLAttributeType::kInt>(name, element_byte_size, value);

      case MLAttributeType::kFloatArray:
        return GetAttributeArrayHelper<MLAttributeType::kFloatArray>(name, element_count, element_byte_size, value);

      case MLAttributeType::kIntArray:
        return GetAttributeArrayHelper<MLAttributeType::kIntArray>(name, element_count, element_byte_size, value);

      default:
        ML_CHECK_BOOL(false);
        break;
    }
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

const std::string* OpKernelInfoWrapper::GetStringAttribute(
    const char* name,
    uint32_t element_index) const {
  // Get the proto attribute
  const AttributeProto* attr = nullptr;
  ML_CHECK_BOOL(impl_->GetAttributeProto(std::string(name), &attr).IsOK());

  // Get the string vector from the attribute
  if (attr->has_s()) {
    return &attr->s();

  } else {
    //  Check the size of the vector
    ML_CHECK_BOOL(attr->strings_size() > 0);
    ML_CHECK_BOOL(element_index < static_cast<uint32_t>(attr->strings_size()));

    return &attr->strings(element_index);
  }
}

ML_API_IMP(OpKernelInfoWrapper::GetStringAttributeElementLength)(
    const char* name,
    uint32_t element_index,
    uint32_t* attribute_element_length) const noexcept {
  try {
    const std::string* protoString = GetStringAttribute(name, element_index);

    // Check for overflow and casting safety
    ML_CHECK_BOOL(protoString->size() < protoString->size() + 1);
    ML_CHECK_BOOL(protoString->size() + 1 <= std::numeric_limits<uint32_t>::max())

    // Set the length including null termination
    *attribute_element_length = static_cast<uint32_t>(protoString->size() + 1);

  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }

  return MLStatus::OK;
}

ML_API_IMP(OpKernelInfoWrapper::GetStringAttributeElement)(
    const char* name,
    uint32_t element_index,
    uint32_t attributeElementSize,
    char* attribute_element) const noexcept {
  try {
    const std::string* protoString = GetStringAttribute(name, element_index);

    size_t stringLength = protoString->size();
    ML_CHECK_BOOL(stringLength < attributeElementSize);
    memcpy(attribute_element, protoString->c_str(), stringLength + 1);

  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }

  return MLStatus::OK;
}

template <MLAttributeType T>
MLStatus OpKernelInfoWrapper::GetAttributeHelper(
    const char* name,
    uint32_t element_byte_size,
    void* value) const {
  typedef typename MLAttributeTypeTraits<T>::Type elementType_t;
  ML_CHECK_BOOL(sizeof(elementType_t) == element_byte_size);
  return ToABIStatus(impl_->GetAttr<typename MLAttributeTypeTraits<T>::Type>(name, static_cast<elementType_t*>(value)));
}

TensorWrapper::TensorWrapper(Tensor* impl) : impl_(impl) {
}

ML_API_IMP(TensorWrapper::GetDimensionCount)(uint32_t* dimensions) const {
  try {
    *dimensions = static_cast<uint32_t>(impl_->Shape().NumDimensions());
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }

  return MLStatus::OK;
}

ML_API_IMP(TensorWrapper::GetDimensions)(
    int64_t* dimensions,
    uint32_t dimension_count) const noexcept {
  try {
    uint32_t count = static_cast<uint32_t>(impl_->Shape().NumDimensions());
    ML_CHECK_BOOL(dimension_count == count);

    for (size_t i = 0; i < dimension_count; ++i) {
      dimensions[i] = impl_->Shape()[i];
    }

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

ML_API_IMP_(MLTensorDataType, TensorWrapper::GetTensorDataType)() const noexcept {
  const DataTypeImpl* type = impl_->DataType();

  if (type == DataTypeImpl::GetType<int32_t>()){
    return MLTypeTraits<int32_t>::TensorType;
  }
  else if (type == DataTypeImpl::GetType<float>()){
    return MLTypeTraits<float>::TensorType;
  }
  else if (type == DataTypeImpl::GetType<uint8_t>()){
    return MLTypeTraits<int32_t>::TensorType;
  }
  else if (type == DataTypeImpl::GetType<uint16_t>()){
    return MLTypeTraits<uint16_t>::TensorType;
  }
  else if (type == DataTypeImpl::GetType<int16_t>()){
    return MLTypeTraits<int16_t>::TensorType;
  }
  else if (type == DataTypeImpl::GetType<int64_t>()){
    return MLTypeTraits<int64_t>::TensorType;
  }
  else if (type == DataTypeImpl::GetType<double>()){
    return MLTypeTraits<double>::TensorType;
  }
  else if (type == DataTypeImpl::GetType<uint32_t>()){
    return MLTypeTraits<uint32_t>::TensorType;
  }
  else if (type == DataTypeImpl::GetType<uint64_t>()){
    return MLTypeTraits<uint64_t>::TensorType;
  }

  // TODO: String and bool tensors.  Lotus bool tensors are not fixed width.

  assert(false);
  return MLTensorDataType::kUndefined;
}

ML_API_IMP_(MLBool, TensorWrapper::IsCPUData)() const noexcept {
  // TODO
  return true;
}

ML_API_IMP_(MLBool, TensorWrapper::IsDataHandle)() const noexcept {
  // TODO
  return false;
}

ML_API_IMP_(void*, TensorWrapper::GetData)() noexcept {
  return impl_->MutableDataRaw();
}

ML_API_IMP_(const void *, TensorWrapper::GetData)() const noexcept {
  return impl_->DataRaw();
}

OpKernelContextWrapper::OpKernelContextWrapper(OpKernelContext* context) : impl_(context) {
  // Pre-size tensor arrays.  Member methods return pointers to these which
  // are stored in these arrays, which would become stale if the vectors reallocate
  // their internal storage.
  inputTensors_.resize(context->InputCount());
  outputTensors_.resize(context->OutputCount());
}

ML_API_IMP_(void *, OpKernelContextWrapper::GetExecutionHandle)() const noexcept {
  return nullptr;
}

ML_API_IMP(OpKernelContextWrapper::GetInputEdgeType)(uint32_t input_index, MLEdgeType* edge_type) const noexcept {
  try {
    // TODO - support non-tensors
    input_index;
    *edge_type = MLEdgeType::kTensor;
    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}
ML_API_IMP(OpKernelContextWrapper::GetOutputEdgeType)(uint32_t output_index, MLEdgeType* edge_type) const noexcept {
  try {
    // TODO - support non-tensors
    output_index;
    *edge_type = MLEdgeType::kTensor;
    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

ML_API_IMP(OpKernelContextWrapper::GetInputTensor)(uint32_t input_index, const IMLOpTensor** tensor) const noexcept {
  try {
    if (inputTensors_[input_index].GetImpl() == nullptr) {
      auto inputTensor = impl_->Input<Tensor>(input_index);
      const_cast<OpKernelContextWrapper*>(this)->inputTensors_[input_index] = const_cast<Tensor*>(inputTensor);
    }

    *tensor = &inputTensors_[input_index];

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

ML_API_IMP(OpKernelContextWrapper::GetOutputTensor)(uint32_t output_index, const int64_t* dimension_sizes, uint32_t dimensions, IMLOpTensor** tensor) noexcept {
  try {
    if (outputTensors_[output_index].GetImpl() == nullptr) {
      TensorShape shape(dimension_sizes, dimensions);
      auto outputTensor = impl_->Output(output_index, shape);
      const_cast<OpKernelContextWrapper*>(this)->outputTensors_[output_index] = outputTensor;
    }

    *tensor = &outputTensors_[output_index];

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

ML_API_IMP_(uint32_t, OpKernelContextWrapper::GetInputCount)() const noexcept {
  return static_cast<uint32_t>(inputTensors_.size());
}

ML_API_IMP_(uint32_t, OpKernelContextWrapper::GetOutputCount)() const noexcept {
  return static_cast<uint32_t>(outputTensors_.size());
}

ML_API_IMP(OpKernelContextWrapper::AllocateTemporaryData)(uint64_t size, void** data) const {
  try {
    *data = nullptr;
    auto& info = impl_->GetAllocatorInfo();
    auto& alloc = AllocatorManager::Instance().GetArena(info.name, info.id);

    *data = alloc.Alloc(size);

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

ML_API_IMP(OpKernelContextWrapper::FreeTemporaryData)(void* data) const {
  try {
    auto& info = impl_->GetAllocatorInfo();
    auto& alloc = AllocatorManager::Instance().GetArena(info.name, info.id);
    if (data) {
      alloc.Free(data);
    }

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

AbiOpKernel::AbiOpKernel(IMLOpKernelCreateFn create_function, const OpKernelInfo& kernelInfo) : OpKernel(kernelInfo), impl_(create_function()) {
  OpKernelInfoWrapper kernelInfoWrapper(&op_kernel_info_);
  ML_CHECK_STATUS(impl_->Initialize(&kernelInfoWrapper));
}

AbiOpKernel::~AbiOpKernel() {
  if (impl_) {
    impl_->Release();
  }
}

Status AbiOpKernel::Compute(OpKernelContext* context) const {
  OpKernelInfoWrapper kernelInfoWrapper(&op_kernel_info_);
  OpKernelContextWrapper kernelContextWrapper(context);
  MLStatus status = impl_->Compute(&kernelInfoWrapper, &kernelContextWrapper);

  if (status != MLStatus::OK)
  {
    return Status(LOTUS, static_cast<StatusCode>(status));
  }

  return Status();
}

}  // namespace Lotus
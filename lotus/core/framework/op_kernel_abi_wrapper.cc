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

// Unreference parameter warning.  TODO - disable once complete
#pragma warning(disable : 4100)

// Disable formatting, which is incorrect for ML_API macros
// _clang-format off

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
  using Type = float;
  static const AttributeProto_AttributeType ProtoType = AttributeProto_AttributeType_FLOAT;
};

template <>
struct MLAttributeTypeTraits<MLAttributeType::kInt> {
  using Type = int64_t;
  static const AttributeProto_AttributeType ProtoType = AttributeProto_AttributeType_INT;
};

template <>
struct MLAttributeTypeTraits<MLAttributeType::kString> {
  static const AttributeProto_AttributeType ProtoType = AttributeProto_AttributeType_STRING;
};

template <MLAttributeType T>
struct MLAttributeArrayTypeTraits {
};

template <>
struct MLAttributeArrayTypeTraits<MLAttributeType::kFloatArray> {
  using Type = float;
  static const AttributeProto_AttributeType ProtoType = AttributeProto_AttributeType_FLOATS;
};

template <>
struct MLAttributeArrayTypeTraits<MLAttributeType::kIntArray> {
  using Type = int64_t;
  static const AttributeProto_AttributeType ProtoType = AttributeProto_AttributeType_INTS;
};

template <>
struct MLAttributeArrayTypeTraits<MLAttributeType::kStringArray> {
  static const AttributeProto_AttributeType ProtoType = AttributeProto_AttributeType_STRINGS;
};

#define ML_ATTR_TO_PROTO_CASE(x) case MLAttributeType::x: return MLAttributeTypeTraits<MLAttributeType::x>::ProtoType;
#define ML_ATTR_VEC_TO_PROTO_CASE(x) case MLAttributeType::x: return MLAttributeArrayTypeTraits<MLAttributeType::x>::ProtoType;

AttributeProto_AttributeType ToProto(MLAttributeType type) {
  switch (type) {
    ML_ATTR_TO_PROTO_CASE(kFloat);
    ML_ATTR_TO_PROTO_CASE(kInt);
    ML_ATTR_TO_PROTO_CASE(kString);

    ML_ATTR_VEC_TO_PROTO_CASE(kFloatArray);
    ML_ATTR_VEC_TO_PROTO_CASE(kIntArray);
    ML_ATTR_VEC_TO_PROTO_CASE(kStringArray);

    default:
      return AttributeProto_AttributeType_UNDEFINED;
  }
}

#define ML_TENSOR_TYPE_CASE(x) if (type == DataTypeImpl::GetType<x>()) { return MLTypeTraits<x>::TensorType; }

::MLTensorDataType ToMLTensorDataType(Lotus::MLDataType type) {
  if (type == DataTypeImpl::GetType<std::string>())
    return MLTensorDataType::kString;

  ML_TENSOR_TYPE_CASE(float);
  ML_TENSOR_TYPE_CASE(uint8_t);
  ML_TENSOR_TYPE_CASE(int8_t);
  ML_TENSOR_TYPE_CASE(uint16_t);
  ML_TENSOR_TYPE_CASE(int16_t);
  ML_TENSOR_TYPE_CASE(int32_t);
  ML_TENSOR_TYPE_CASE(int64_t);
  ML_TENSOR_TYPE_CASE(bool);
  ML_TENSOR_TYPE_CASE(double);
  ML_TENSOR_TYPE_CASE(uint32_t);
  ML_TENSOR_TYPE_CASE(uint64_t);
  ML_TENSOR_TYPE_CASE(MLFloat16);

  // TODO - non-primitive traits classes: string, float16, complex64, complex128
  ML_CHECK_STATUS(MLStatus::NOT_IMPLEMENTED);
}

#undef ML_TENSOR_TYPE_CASE
#define ML_TENSOR_TYPE_CASE(x) if (type == MLTypeTraits<x>::TensorType) { return DataTypeImpl::GetTensorType<x>(); }

Lotus::MLDataType ToTensorDataType(::MLTensorDataType type) {
  if (type == MLTensorDataType::kString)
    return DataTypeImpl::GetTensorType<std::string>();

  ML_TENSOR_TYPE_CASE(float);
  ML_TENSOR_TYPE_CASE(uint8_t);
  ML_TENSOR_TYPE_CASE(int8_t);
  ML_TENSOR_TYPE_CASE(uint16_t);
  ML_TENSOR_TYPE_CASE(int16_t);
  ML_TENSOR_TYPE_CASE(int32_t);
  ML_TENSOR_TYPE_CASE(int64_t);
  ML_TENSOR_TYPE_CASE(bool);
  ML_TENSOR_TYPE_CASE(double);
  ML_TENSOR_TYPE_CASE(uint32_t);
  ML_TENSOR_TYPE_CASE(uint64_t);
  ML_TENSOR_TYPE_CASE(MLFloat16);

  // TODO - non-primitive traits classes: string, float16, complex64, complex128

  ML_CHECK_STATUS(MLStatus::NOT_IMPLEMENTED);
}

::MLTensorDataType ToMLTensorDataType(TensorProto_DataType type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return MLTensorDataType::kFloat;

    case TensorProto_DataType_UINT8:
      return MLTensorDataType::kUInt8;

    case TensorProto_DataType_INT8:
      return MLTensorDataType::kInt8;

    case TensorProto_DataType_UINT16:
      return MLTensorDataType::kUInt16;

    case TensorProto_DataType_INT16:
      return MLTensorDataType::kInt16;

    case TensorProto_DataType_INT32:
      return MLTensorDataType::kInt32;

    case TensorProto_DataType_INT64:
      return MLTensorDataType::kInt64;

    case TensorProto_DataType_STRING:
      return MLTensorDataType::kString;

    case TensorProto_DataType_BOOL:
      return MLTensorDataType::kBool;

    case TensorProto_DataType_FLOAT16:
      return MLTensorDataType::kFloat16;

    case TensorProto_DataType_DOUBLE:
      return MLTensorDataType::kDouble;

    case TensorProto_DataType_UINT32:
      return MLTensorDataType::kUInt32;

    case TensorProto_DataType_UINT64:
      return MLTensorDataType::kUInt64;

    case TensorProto_DataType_COMPLEX64:
      return MLTensorDataType::kComplex64;

    case TensorProto_DataType_COMPLEX128:
      return MLTensorDataType::kComplex128;

    default:
      ML_CHECK_STATUS(MLStatus::NOT_IMPLEMENTED);
  }
}
::MLEdgeType ToMLEdgeType(const TypeProto* type) {
  // Initialized to undefined class and data type
  MLEdgeType ret = {};

  ML_CHECK_BOOL(type->value_case() == TypeProto::kTensorType ||
                type->value_case() == TypeProto::VALUE_NOT_SET);  

  if (type->value_case() == TypeProto::kTensorType) {    
    ret.edge_class = MLEdgeClass::kTensor;
    const TypeProto_Tensor tensor_type = type->tensor_type();
    if (tensor_type.has_elem_type()) {
      ret.tensor_data_type = ToMLTensorDataType(tensor_type.elem_type());
    }
  }

  // TODO support non-tensor types

  return ret;
}

std::string ToTypeString(MLEdgeType type) {
  // TODO - handle non-tensor types
  if (type.edge_class != MLEdgeClass::kTensor) {
    ML_CHECK_STATUS(MLStatus::NOT_IMPLEMENTED);
  }

  switch (type.tensor_data_type) {
    case MLTensorDataType::kFloat:
      return "tensor(float)";

    case MLTensorDataType::kUInt8:
      return "tensor(uint8)";

    case MLTensorDataType::kInt8:
      return "tensor(int8)";

    case MLTensorDataType::kUInt16:
      return "tensor(uint16)";

    case MLTensorDataType::kInt16:
      return "tensor(int16)";

    case MLTensorDataType::kInt32:
      return "tensor(int32)";

    case MLTensorDataType::kInt64:
      return "tensor(int64)";

    case MLTensorDataType::kString:
      return "tensor(string)";

    case MLTensorDataType::kBool:
      return "tensor(bool)";

    case MLTensorDataType::kFloat16:
      return "tensor(float16)";

    case MLTensorDataType::kDouble:
      return "tensor(double)";

    case MLTensorDataType::kUInt32:
      return "tensor(uint32)";

    case MLTensorDataType::kUInt64:
      return "tensor(uint64)";

    case MLTensorDataType::kComplex64:
      return "tensor(complext64)";

    case MLTensorDataType::kComplex128:
      return "tensor(complext128)";

    default:
      ML_CHECK_STATUS(MLStatus::NOT_IMPLEMENTED);
  }
}

OpKernelInfoWrapper::OpKernelInfoWrapper(const OpKernelInfo* kernel_info) : impl_(kernel_info), OpNodeInfoWrapper(kernel_info) {
}

// Prevents the templatized class name being parsed as multiple arguments to a macro
#define NODEINFO_WRAPPER_CLASS OpNodeInfoWrapper<Base_t, NodeInfoImpl_t>

template<class Base_t, class NodeInfoImpl_t>
ML_API_IMP(NODEINFO_WRAPPER_CLASS::GetAttributeElementCount)(
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

template<class Base_t, class NodeInfoImpl_t>
template <MLAttributeType T>
MLStatus OpNodeInfoWrapper<Base_t, NodeInfoImpl_t>::GetAttributeArrayHelper(
    const char* name,
    uint32_t element_count,
    uint32_t element_byte_size,
    void* values) const {
  typedef typename MLAttributeArrayTypeTraits<T>::Type elementType_t;
  ML_CHECK_BOOL(sizeof(elementType_t) == element_byte_size);

  ML_CHECK_STATUS(ToABIStatus(impl_->GetAttrs(name, gsl::span<elementType_t>(static_cast<typename MLAttributeArrayTypeTraits<T>::Type*>(values), element_count))));
  return MLStatus::OK;
}

template<class Base_t, class NodeInfoImpl_t>
ML_API_IMP(NODEINFO_WRAPPER_CLASS::GetAttribute)(
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

template<class Base_t, class NodeInfoImpl_t>
const std::string* OpNodeInfoWrapper<Base_t, NodeInfoImpl_t>::GetStringAttribute(
    const char* name,
    uint32_t element_index) const {
  // Get the proto attribute
  const AttributeProto* attr = impl_->GetAttribute(std::string(name));

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

template<class Base_t, class NodeInfoImpl_t>
ML_API_IMP(NODEINFO_WRAPPER_CLASS::GetStringAttributeElementLength)(
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

template<class Base_t, class NodeInfoImpl_t>
ML_API_IMP(NODEINFO_WRAPPER_CLASS::GetStringAttributeElement)(
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

template<class Base_t, class NodeInfoImpl_t>
template <MLAttributeType T>
MLStatus OpNodeInfoWrapper<Base_t, NodeInfoImpl_t>::GetAttributeHelper(
    const char* name,
    uint32_t element_byte_size,
    void* value) const {
  typedef typename MLAttributeTypeTraits<T>::Type elementType_t;
  ML_CHECK_BOOL(sizeof(elementType_t) == element_byte_size);
  return ToABIStatus(impl_->template GetAttr<elementType_t>(name, static_cast<elementType_t*>(value)));
}

template<class Base_t, class NodeInfoImpl_t>
ML_API_IMP(NODEINFO_WRAPPER_CLASS::GetInputEdgeType) (uint32_t input_index, MLEdgeType* edge_type) const noexcept {
  try {
    const TypeProto* type = impl_->GetInputType(input_index);
    ML_CHECK_BOOL(type != nullptr);
    *edge_type = ToMLEdgeType(type);

    assert(edge_type->edge_class != MLEdgeClass::kUndefined);
    assert((edge_type->edge_class != MLEdgeClass::kTensor && edge_type->edge_class != MLEdgeClass::kTensorSequence) ||
                  edge_type->tensor_data_type != MLTensorDataType::kUndefined);

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

template<class Base_t, class NodeInfoImpl_t>
ML_API_IMP(NODEINFO_WRAPPER_CLASS::GetOutputEdgeType) (uint32_t output_index, MLEdgeType* edge_type) const noexcept {
  try {
    const TypeProto* type = impl_->GetOutputType(output_index);
    ML_CHECK_BOOL(type != nullptr);
    *edge_type = ToMLEdgeType(type);

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

ML_API_IMP_(bool, OpKernelInfoWrapper::HasTensorShapeInfo)() const noexcept {
  return false;
}

ML_API_IMP(OpKernelInfoWrapper::GetTensorShapeInfo) (const IMLOpKernelTensorShapeInfo** shapeInfo) const noexcept {
  if (!HasTensorShapeInfo()) {
      *shapeInfo = nullptr;
      return MLStatus::FAIL;
  }

  *shapeInfo = this;
  return MLStatus::OK;
}

ML_API_IMP_(const void*, OpKernelInfoWrapper::GetExecutionHandle)() const noexcept {
  const IExecutionProvider* executionProvider = impl_->GetExecutionProvider();
  return executionProvider->GetExecutionHandle();
}

template<class Base_t, class NodeInfoImpl_t>
ML_API_IMP_(uint32_t, NODEINFO_WRAPPER_CLASS::GetInputCount)() const noexcept {
  return impl_->GetInputCount();
}

template<class Base_t, class NodeInfoImpl_t>
ML_API_IMP_(uint32_t, NODEINFO_WRAPPER_CLASS::GetOutputCount)() const noexcept {
  return impl_->GetOutputCount();
}

ML_API_IMP(OpKernelInfoWrapper::GetInputTensorDimensionCount)(uint32_t input_index, uint32_t* dimension_count) const noexcept {
  // TODO
  return MLStatus::NOT_IMPLEMENTED;
}

ML_API_IMP(OpKernelInfoWrapper::GetInputTensorShape)(uint32_t input_index, uint32_t dimension_count, int64_t* dimensions) const noexcept {
  // TODO
  return MLStatus::NOT_IMPLEMENTED;
}

ML_API_IMP_(bool, OpKernelInfoWrapper::HasOutputShapeInfo)() const noexcept {
  // TODO
  return false;
}

ML_API_IMP(OpKernelInfoWrapper::GetOutputTensorDimensionCount)(uint32_t output_index, uint32_t* dimension_count) const noexcept {
  return MLStatus::NOT_IMPLEMENTED;
}

ML_API_IMP(OpKernelInfoWrapper::GetOutputTensorShape)(uint32_t output_index, uint32_t dimension_count, int64_t* dimensions) const noexcept {
  return MLStatus::NOT_IMPLEMENTED;
}

TensorWrapper::TensorWrapper(Tensor* impl) : impl_(impl) {
}

ML_API_IMP_(uint32_t, TensorWrapper::GetDimensionCount)() const noexcept{
  return gsl::narrow_cast<uint32_t>(impl_->Shape().NumDimensions());
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
  try {
    return ToMLTensorDataType(impl_->DataType());
  } catch (...) {
    return MLTensorDataType::kUndefined;
  }
}

ML_API_IMP_(bool, TensorWrapper::IsCPUData)() const noexcept {
  // tells DML whether this tensor is in CPU memory
  return impl_->Location().name == CPU || impl_->Location().mem_type == kMemTypeCPU;
}

ML_API_IMP_(bool, TensorWrapper::IsDataHandle)() const noexcept {
  // tells DML whether this tensor is in DML device memory
  // TODO: change to Location().name == DML once DML provider is in
  return !IsCPUData();
}

ML_API_IMP_(void*, TensorWrapper::GetData)() noexcept {
  return impl_->MutableDataRaw();
}

ML_API_IMP_(const void*, TensorWrapper::GetData)() const noexcept {
  return impl_->DataRaw();
}

OpKernelContextWrapper::OpKernelContextWrapper(OpKernelContext* context, const IExecutionProvider* provider) : impl_(context), provider_(provider) {
  // Pre-size tensor arrays.  Member methods return pointers to these which
  // are stored in these arrays, which would become stale if the vectors reallocate
  // their internal storage.
  inputTensors_.resize(context->InputCount());
  outputTensors_.resize(context->OutputCount());
}

ML_API_IMP(OpKernelContextWrapper::GetInputTensor)(uint32_t input_index, const IMLOpTensor** tensor) const noexcept {
  *tensor = nullptr;

  try {
    if (inputTensors_[input_index].GetInterface() == nullptr) {
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

ML_API_IMP(OpKernelContextWrapper::GetOutputTensor)(uint32_t output_index, IMLOpTensor** tensor) noexcept {
  *tensor = nullptr;

  // TODO implement this for shaped kernels
  return MLStatus::NOT_IMPLEMENTED;
}

ML_API_IMP(OpKernelContextWrapper::GetDynamicOutputTensor)(uint32_t output_index, const int64_t* dimension_sizes, uint32_t dimensions, IMLOpTensor** tensor) noexcept {
  *tensor = nullptr;

  try {
    if (outputTensors_[output_index].GetInterface() == nullptr) {
      TensorShape shape(dimension_sizes, dimensions);
      auto outputTensor = impl_->Output(output_index, shape);
      if (outputTensor)
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

ML_API_IMP(OpKernelContextWrapper::AllocateTemporaryData)(uint64_t size, void** data) const {
  try {
    *data = nullptr;
    AllocatorPtr alloc;
    ML_CHECK_STATUS(ToABIStatus(impl_->GetTempSpaceAllocator(&alloc)));

    *data = alloc->Alloc(size);

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

ML_API_IMP(OpKernelContextWrapper::FreeTemporaryData)(void* data) const {
  try {
    AllocatorPtr alloc;
    ML_CHECK_STATUS(ToABIStatus(impl_->GetTempSpaceAllocator(&alloc)));
    if (data) {
      alloc->Free(data);
    }

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

ML_API_IMP_(const void*, OpKernelContextWrapper::GetExecutionHandle)() const noexcept {
  return provider_;
}

AbiOpKernel::AbiOpKernel(IMLOpKernelCreateFn create_function, const OpKernelInfo& kernel_info) : OpKernel(kernel_info) {
  OpKernelInfoWrapper kernelInfoWrapper(&op_kernel_info_);
  ML_CHECK_STATUS(create_function(kernelInfoWrapper, &impl_));
}

AbiOpKernel::~AbiOpKernel() {
  if (impl_) {
    impl_->Release();
  }
}

Status AbiOpKernel::Compute(OpKernelContext* context) const {
  OpKernelInfoWrapper kernelInfoWrapper(&op_kernel_info_);
  OpKernelContextWrapper kernelContextWrapper(context, op_kernel_info_.GetExecutionProvider());
  MLStatus status = impl_->Compute(&kernelContextWrapper);

  if (status != MLStatus::OK) {
    return Status(LOTUS, static_cast<StatusCode>(status));
  }

  return Status();
}

AbiCustomRegistry::AbiCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) : custom_registry_(custom_registry) {
}

OpSchema::FormalParameterOption AbiCustomRegistry::ConvertFormalParameterOption(MLFormalParameterOptions options) {
  switch (options) {
    case MLFormalParameterOptions::kSingle:
      return OpSchema::FormalParameterOption::Single;

    case MLFormalParameterOptions::kOptional:
      return OpSchema::FormalParameterOption::Optional;

    case MLFormalParameterOptions::kVariadic:
      return OpSchema::FormalParameterOption::Variadic;

    default:
      ML_CHECK_STATUS(MLStatus::NOT_IMPLEMENTED);
  }
}

// Convert edge types from the ABI types to ONNX strings
std::string AbiCustomRegistry::ConvertFormalParameterType(const MLFormalParameter& formal_parameter) {
  ML_CHECK_BOOL(formal_parameter.type_format == MLFormalParameterTypeFormat::kLabel || 
                formal_parameter.type_format == MLFormalParameterTypeFormat::kEdgeType);

  if (formal_parameter.type_format == MLFormalParameterTypeFormat::kLabel) {
    return formal_parameter.type_label;
  } else {
    return ToTypeString(formal_parameter.edge_type);
  }
}

// Convert type constraints from the ABI types to ONNX strings
std::vector<std::string> ConvertTypeConstraintTypes(const MLTypeConstraint& constraint) {
  std::vector<std::string> ret;
  ret.reserve(constraint.allowed_type_count);

  for (uint32_t i = 0; i < constraint.allowed_type_count; ++i) {
    ret.emplace_back(ToTypeString(constraint.allowed_types[i]));
  }

  return ret;
}

// Convert attributes and defaults from the ABI to ONNX schema
void AbiCustomRegistry::SetAttributesAndDefaults(OpSchema& schema, const MLSchemaDefinition& abi_schema) {
  // Create a map with default attributes
  std::map<std::string, const MLAttributeNameValue*> default_attributes;
  for (uint32 attribute_index = 0; attribute_index < abi_schema.default_attribute_count; ++attribute_index) {
    const MLAttributeNameValue& default_attribute = abi_schema.default_attributes[attribute_index];
    default_attributes[default_attribute.name] = &default_attribute;
  }

  // Set each attribute along with default values, looked up by name, if available
  for (uint32 attribute_index = 0; attribute_index < abi_schema.attribute_count; ++attribute_index) {
    const MLAttribute& attribute = abi_schema.attributes[attribute_index];
    auto default_val = default_attributes.find(attribute.name);
    if (default_val == default_attributes.end()) {
      schema.Attr(attribute.name, "", ToProto(attribute.type), attribute.required);
    } else {
      ML_CHECK_BOOL(!attribute.required);
      ML_CHECK_BOOL(attribute.type == default_val->second->type);
      uint32_t default_count = default_val->second->value_count;

      switch (attribute.type) {
        case MLAttributeType::kFloat:
          ML_CHECK_BOOL(default_count == 1);
          schema.Attr(attribute.name, "", ToProto(attribute.type), default_val->second->floats[0]);
          break;

        case MLAttributeType::kInt:
          ML_CHECK_BOOL(default_count == 1);
          schema.Attr(attribute.name, "", ToProto(attribute.type), default_val->second->ints[0]);
          break;

        case MLAttributeType::kString:
          ML_CHECK_BOOL(default_count == 1);
          schema.Attr(attribute.name, "", ToProto(attribute.type), std::string(default_val->second->strings[0]));
          break;

        case MLAttributeType::kFloatArray: {
          std::vector<float> default_vals(default_val->second->floats, default_val->second->floats + default_count);
          schema.Attr(attribute.name, "", ToProto(attribute.type), default_vals);
          break;
        }

        case MLAttributeType::kIntArray: {
          std::vector<int64_t> default_vals(default_val->second->ints, default_val->second->ints + default_count);
          schema.Attr(attribute.name, "", ToProto(attribute.type), default_vals);
          break;
        }

        case MLAttributeType::kStringArray: {
          std::vector<std::string> default_vals(default_val->second->strings, default_val->second->strings + default_count);
          schema.Attr(attribute.name, "", ToProto(attribute.type), default_vals);
          break;
        }

        default:
          ML_CHECK_BOOL(false);
          break;
      }

      // Remove the default attribute from the map, to later ensure defaults matched attributes
      default_attributes.erase(attribute.name);
    }
  }

  ML_CHECK_BOOL(default_attributes.empty());
}

// Convert a schema from the ABI to ONNX type
OpSchema AbiCustomRegistry::ConvertOpSchema(const char* domain, const MLSchemaDefinition& abi_schema) {
  // Set the op schema name, domain, and version
  OpSchema schema(abi_schema.name, "", 0);
  schema.SetDomain(domain);
  schema.SinceVersion(abi_schema.operator_set_since_version);

  // ONNX fails if using an empty string for edge names, although their names don't
  // matter for us.
  const char* empty_name = " ";

  // Populate inputs
  for (uint32 input_index = 0; input_index < abi_schema.input_count; ++input_index) {
    schema.Input(
        input_index,
        empty_name,
        "",
        ConvertFormalParameterType(abi_schema.inputs[input_index]),
        ConvertFormalParameterOption(abi_schema.inputs[input_index].options));
  }

  // Populate outputs
  for (uint32 output_index = 0; output_index < abi_schema.output_count; ++output_index) {
    schema.Output(
        output_index,
        empty_name,
        "",
        ConvertFormalParameterType(abi_schema.outputs[output_index]),
        ConvertFormalParameterOption(abi_schema.outputs[output_index].options));
  }

  // Populate type constraints
  for (uint32 constraint_index = 0; constraint_index < abi_schema.type_constraint_count; ++constraint_index) {
    schema.TypeConstraint(
        abi_schema.type_constraints[constraint_index].type_label,
        ConvertTypeConstraintTypes(abi_schema.type_constraints[constraint_index]),
        "");
  }

  // Set attribute defaults
  SetAttributesAndDefaults(schema, abi_schema);

  auto type_inference_func = abi_schema.type_inference_function;
  auto shape_inference_func = abi_schema.shape_inference_function;

  // Set an inferencing method
  if (shape_inference_func || type_inference_func) {
    schema.TypeAndShapeInferenceFunction([=](InferenceContext& ctx) {
      OpNodeProtoHelper<InferenceContext> node_info(&ctx);
      MLInferenceContext abi_context(&node_info, &ctx);

      // Do type inference
      if (type_inference_func) {
        (*type_inference_func)(abi_schema.type_inference_function_context, &abi_context);
      }

      // Do shape inference if all input tensor shapes are known
      if (shape_inference_func && InputTensorShapesDefined(abi_context)) {
        (*shape_inference_func)(abi_schema.shape_inference_function_context, &abi_context);
      }
    });
  }

  return schema;
}
  
// Static method querying whether tensor shapes are defined, during wrappers 
// of shape inference callbacks.
bool AbiCustomRegistry::InputTensorShapesDefined(MLInferenceContext& abi_context) {
  MLShapeInferenceContext context_wrapper(&abi_context);
  uint32_t input_count = context_wrapper.GetInputCount();

  for (uint32_t input_index = 0; input_index < input_count; ++input_index) {
    MLEdgeType edge_type = context_wrapper.GetInputEdgeType(input_index);

    if (edge_type.edge_class == MLEdgeClass::kTensor) {
      uint32_t input_dim_count = context_wrapper.GetInputTensorDimensionCount(input_index);

      for (uint32_t input_dim = 0; input_dim < input_dim_count; ++input_dim) {
        if (!abi_context.GetContext()->getInputType(input_dim)->tensor_type().shape().dim(input_dim).has_dim_value()) {
          return false;
        }
      }
    }
  }

  return true;
}

ML_API_IMP(AbiCustomRegistry::RegisterOpSetFromSchema)(
    const MLOperatorSetId* opSetId,
    int baseline_version,
    const MLSchemaDefinition* const* schema,
    uint32_t schema_count) const noexcept {
  try {
    // TODO - handle baseline_version;

    std::vector<OpSchema> schema_vector;
    schema_vector.reserve(schema_count);

    // Convert schema to ONNX types and accumulate them in a vector
    for (uint32 i = 0; i < schema_count; ++i) {
      schema_vector.emplace_back(ConvertOpSchema(opSetId->domain, *schema[i]));
    }

    // Register the operator set with Lotus
    LOTUS_ENFORCE(custom_registry_->RegisterCustomOpSet(schema_vector, opSetId->domain, opSetId->version).IsOK());

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

ML_API_IMP(AbiCustomRegistry::RegisterOpKernel)(
    const MLOpKernelDefinition* op_kernel,
    MLOpKernelOptions options,
    IMLOpKernelCreateFn op_kernel_factory) const noexcept {
  try {
    // Set the name, domain, version, and provider
    KernelDefBuilder builder(op_kernel->name);
    builder.Domain(op_kernel->domain)
        .SinceVersion(op_kernel->operator_set_since_version)
        .Provider(op_kernel->execution_provider_name);

    // Set type constraints
    for (uint32_t i = 0; i < op_kernel->type_constraint_count; ++i) {
      std::vector<MLDataType> types;
      types.reserve(op_kernel->type_constraints[i].allowed_type_count);

      for (uint32_t j = 0; j < op_kernel->type_constraints[i].allowed_type_count; ++j) {
        // TODO - handle non-tensor types
        if (op_kernel->type_constraints[i].allowed_types[j].edge_class != MLEdgeClass::kTensor) {
          ML_CHECK_STATUS(MLStatus::NOT_IMPLEMENTED);
        }

        types.push_back(ToTensorDataType(op_kernel->type_constraints[i].allowed_types[j].tensor_data_type));
      }

      builder.TypeConstraint(op_kernel->type_constraints[i].type_label, types);
    }

    // TODO - handle default attributes, shape inference, and options
    custom_registry_->RegisterCustomKernel(builder,
                                           [op_kernel_factory](const OpKernelInfo& info) -> OpKernel* { return new Lotus::AbiOpKernel(op_kernel_factory, info); });

    return MLStatus::OK;
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }
}

ML_API_IMP(MLInferenceContext::GetInputTensorDimensionCount)(uint32_t input_index, uint32_t* dimension_count) const noexcept{
  try {
    LOTUS_ENFORCE(context_->getInputType(input_index)->has_tensor_type());
    *dimension_count = context_->getInputType(input_index)->tensor_type().shape().dim_size();
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }

  return MLStatus::OK;
}

ML_API_IMP(MLInferenceContext::GetInputTensorShape)(uint32_t input_index, uint32_t dimension_count, int64_t* dimensions) const noexcept{
  try {
    LOTUS_ENFORCE(context_->getInputType(input_index)->has_tensor_type());
    ML_CHECK_BOOL(static_cast<size_t>(dimension_count) == context_->getInputType(input_index)->tensor_type().shape().dim_size());

    for (uint32_t i = 0; i < dimension_count; ++i) {
      // Shape inference is only done when all dimensions of all inputs have known values.
      // This avoids potentially dangerous changes in behavior to external inference functions 
      // based on implementation details of upstream nodes
      assert(context_->getInputType(input_index)->tensor_type().shape().dim(i).has_dim_value());
      dimensions[i] = context_->getInputType(input_index)->tensor_type().shape().dim(i).dim_value();
    }

  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }

  return MLStatus::OK;
}

ML_API_IMP(MLInferenceContext::SetOutputTensorShape)(uint32_t output_index, uint32_t dimension_count, const int64_t* dimensions) noexcept{
  try {
    MLEdgeType edge_type;
    ML_CHECK_STATUS(GetOutputEdgeType(output_index, &edge_type));
    ML_CHECK_BOOL(edge_type.edge_class == MLEdgeClass::kUndefined || edge_type.edge_class == MLEdgeClass::kTensor);

    for (uint32_t i = 0; i < dimension_count; ++i) {
      ML_CHECK_BOOL(dimensions[i] > 0);

      // In the process of calling mutable_tensor_type, the type may switch from undefined to tensor
      auto dim = context_->getOutputType(output_index)->mutable_tensor_type()->mutable_shape()->add_dim();
      dim->set_dim_value(dimensions[i]);
    }
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }

  return MLStatus::OK;
}

ML_API_IMP(MLInferenceContext::SetOutputEdgeType)(uint32_t output_index, const MLEdgeType* edge_type) const noexcept{
  try {
    std::string type_str = ToTypeString(*edge_type);
    context_->getOutputType(output_index)->CopyFrom(Utils::DataTypeUtils::ToTypeProto(&type_str));
  } catch (const MLStatusException& ex) {
    return ex.GetStatus();
  } catch (...) {
    return MLStatus::FAIL;
  }

  return MLStatus::OK;
}
}  // namespace Lotus

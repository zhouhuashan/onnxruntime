//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#pragma once

#include "core/inc/op_kernel_author.h"

// Disable formatting, which is incorrect for ML_API macros
// clang-format off

namespace Lotus {

// Encapsulation of shapes across different edges of an operator.  Non-tensor
// edges and unused edges have an empty array of dimensions.
class EdgeShapes {
 public:
  EdgeShapes() = default;

  EdgeShapes(size_t count) : shapes_(count) {}

  const std::vector<int64_t>& GetShape(size_t input_index) const {
    return shapes_[input_index];
  }

  std::vector<int64_t>& GetMutableShape(size_t input_index) {
    return shapes_[input_index];
  }

  size_t EdgeCount() const { return shapes_.size(); }

  void Reset(size_t edge_count) {
    shapes_.clear();
    shapes_.resize(edge_count);
  }

  bool operator!=(const EdgeShapes& other) const noexcept {
    return (shapes_ != other.shapes_);
  }

 private:
  std::vector<std::vector<int64_t>> shapes_;
};

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
class OpNodeInfoWrapper : public Base1_t, public Base2_t {
 public:
  OpNodeInfoWrapper() = delete;
  OpNodeInfoWrapper(const OpNodeProtoHelper<NodeInfoImpl_t>* impl, const EdgeShapes* input_shapes_override) : impl_(impl), input_shapes_override_(input_shapes_override) {}

  ML_API_IMP(GetAttributeElementCount)(
      MLAttributeType type,
      const char* name,
      uint32_t* element_count) const noexcept override;

  template <MLAttributeType T>
  MLStatus GetAttributeArrayHelper(
      const char* name,
      uint32_t element_count,
      uint32_t element_byte_size,
      void* values) const;

  ML_API_IMP(GetAttribute)(
      const char* name,
      MLAttributeType type,
      uint32_t element_count,
      uint32_t element_byte_size,
      void* value) const noexcept override;

  ML_API_IMP(GetStringAttributeElementLength)(
      const char* name,
      uint32_t element_index,
      uint32_t* attribute_element_length) const noexcept override;

  ML_API_IMP(GetStringAttributeElement)(
      const char* name,
      uint32_t element_index,
      uint32_t attribute_element_length,
      char* attribute_element) const noexcept override;

  ML_API_IMP_(uint32_t, GetInputCount)() const noexcept override;
  ML_API_IMP_(uint32_t, GetOutputCount)() const noexcept override;

  ML_API_IMP(GetInputEdgeType)(uint32_t input_index, MLEdgeType* edge_type) const noexcept override;
  ML_API_IMP(GetOutputEdgeType)(uint32_t output_index, MLEdgeType* edge_type) const noexcept;

  ML_API_IMP(GetInputTensorDimensionCount)(uint32_t input_index, uint32_t* dimension_count) const noexcept;
  ML_API_IMP(GetInputTensorShape)(uint32_t input_index, uint32_t dimension_count, int64_t* dimensions) const noexcept;

 protected:
  // Lifetime is managed by the caller and guaranteed to outlive this class
  const OpNodeProtoHelper<NodeInfoImpl_t>* impl_ = nullptr;

 private:
  template <MLAttributeType T>
  MLStatus GetAttributeHelper(
      const char* name,
      uint32_t element_byte_size,
      void* value) const;

  const std::string* GetStringAttribute(
      const char* name,
      uint32_t element_index) const;

  // May be null
  const EdgeShapes* input_shapes_override_;
};

class OpKernelInfoWrapper : public OpNodeInfoWrapper<ProtoHelperNodeContext, IMLOpKernelInfo, IMLOpKernelTensorShapeInfo> {
 public:
  OpKernelInfoWrapper(
      const OpKernelInfo* kernel_info,
      const EdgeShapes* input_shape_overrides,
      const EdgeShapes* inferred_output_shapes,
      bool allow_input_shape_query,
      bool allow_output_shape_query);

  // HasTensorShapeInfo returns false if and only if the kernel is registered using
  // MLOpKernelOptions::kAllowDynamicInputTensorSizes.  If this flag is specified and upstream
  // shapes are known when the kernel is created, HasTensorShapeInfo still returns false.
  ML_API_IMP_(bool, HasTensorShapeInfo)() const noexcept override;
  ML_API_IMP(GetTensorShapeInfo)(const IMLOpKernelTensorShapeInfo** shapeInfo) const noexcept override;

  ML_API_IMP_(const void*, GetExecutionHandle)() const noexcept override;

  // IMLOpKernelTensorShapeInfo methods.
  ML_API_IMP(GetOutputTensorDimensionCount)(uint32_t input_index, uint32_t* dimension_count) const noexcept;
  ML_API_IMP_(bool, HasOutputShapeInfo)() const noexcept override;
  ML_API_IMP(GetOutputTensorShape)(uint32_t input_index, uint32_t dimension_count, int64_t* dimensions) const noexcept;

  // For shape info, in addition to the info
  const EdgeShapes* inferred_output_shapes_ = nullptr;
  bool allow_input_shape_query_ = false;
  bool allow_output_shape_query_ = false;

  const OpKernelInfo* impl_ = nullptr;
};

class TensorWrapper : public IMLOpTensor {
 public:
  TensorWrapper() = default;

  TensorWrapper(Tensor* impl);

  ML_API_IMP_(uint32_t, GetDimensionCount)() const noexcept override;

  ML_API_IMP(GetDimensions)(
      int64_t* dimensions,
      uint32_t dimension_count) const noexcept override;

  ML_API_IMP_(MLTensorDataType, GetTensorDataType)() const noexcept override;

  ML_API_IMP_(bool, IsCPUData)() const noexcept override;

  ML_API_IMP_(bool, IsDataHandle)() const noexcept override;

  ML_API_IMP_(void*, GetData)() noexcept override;

  ML_API_IMP_(const void*, GetData)() const noexcept override;

  ML_API_(bool, IsUnused)() const noexcept override {
    return impl_ == nullptr;
  }

  const Tensor* GetInterface() const { return nullptr; }
  Tensor* GetInterface() { return nullptr; }

 private:
  // Lifetime is managed by the caller and guaranteed to outlive this class
  Tensor* impl_ = nullptr;
};

class OpKernelContextWrapper : public IMLOpKernelContext {
 public:
  OpKernelContextWrapper(OpKernelContext* context, const IExecutionProvider* provider, const EdgeShapes* output_shapes);

  ML_API_IMP(GetInputTensor)(uint32_t input_index, const IMLOpTensor** tensor) const noexcept override;
  ML_API_IMP(GetOutputTensor)(uint32_t output_index, IMLOpTensor** tensor) noexcept override;
  ML_API_IMP(GetOutputTensor)(uint32_t output_index, const int64_t* dimension_sizes, uint32_t dimensions, IMLOpTensor** tensor) noexcept override;

  ML_API_IMP(AllocateTemporaryData)(uint64_t size, void** data) const override;
  ML_API_IMP(FreeTemporaryData)(void* data) const override;

  ML_API_IMP_(const void*, GetExecutionHandle)() const noexcept override;

 protected:
  // Lifetime is managed by the caller and guaranteed to outlive this class
  OpKernelContext* impl_ = nullptr;
  const EdgeShapes* output_shapes_ = nullptr;

  // TODO - Use a custom STL allocator to avoid heap allocations in the common case.
  std::vector<TensorWrapper> input_tensors_;
  std::vector<TensorWrapper> output_tensors_;

  const IExecutionProvider* provider_ = nullptr;
};

class AbiOpKernel : public OpKernel {
 public:
  AbiOpKernel(
      IMLOpKernelCreateFn create_function,
      const OpKernelInfo& kernel_info,
      bool requires_input_shapes_at_creation,
      bool requires_output_shapes_at_creation,
      MLShapeInferenceFunction shape_inference_function,
      void* shape_inference_function_context);

  ~AbiOpKernel();

  Status Compute(OpKernelContext* context) const override;

 protected:
  bool RequiresLazyInitialization() const { return (create_function_ != nullptr) && !lazy_initialized_; };
  void SetLazyInitialized() { lazy_initialized_ = true; };

  EdgeShapes GetInputShapes(OpKernelContext* context) const;

  bool InputTensorShapesDefined() const;
  bool InputSizesInferencedFromSchema() const;
  void InferAndVerifyOutputSizes(const EdgeShapes* input_shapes, EdgeShapes& output_shapes) const;
  bool requires_input_shapes_at_creation_ = false;
  bool requires_output_shapes_at_creation_ = false;

  // Lifetime is managed by the caller and guaranteed to outlive this class
  IMLOpKernel* impl_ = nullptr;

  // This is null unless the kernel requires lazy initialization
  IMLOpKernelCreateFn create_function_ = nullptr;
  volatile bool lazy_initialized_ = false;

  MLShapeInferenceFunction shape_inference_function_;
  void* shape_inference_function_context_;

  std::mutex mutex_;
  EdgeShapes input_shapes_of_kernel_inference_;
  EdgeShapes inferred_output_shapes_;
};

class MLSchemaInferenceContext final : public OpNodeInfoWrapper<InferenceContext, IMLShapeInferenceContext, IMLTypeInferenceContext> {
 public:
  MLSchemaInferenceContext() = delete;
  MLSchemaInferenceContext(OpNodeProtoHelper<InferenceContext>* info, InferenceContext* ctx) : OpNodeInfoWrapper(info, nullptr), context_(ctx) {}

  InferenceContext* GetContext() const {
    return context_;
  }

  ML_API_IMP(SetOutputEdgeType)(uint32_t output_index, const MLEdgeType* edge_type) const noexcept override;
  ML_API_IMP(SetOutputTensorShape)(uint32_t output_index, uint32_t dimension_count, const int64_t* dimensions) noexcept override;

 private:
  InferenceContext* context_ = nullptr;
};

class MLKernelInferenceContext final : public OpNodeInfoWrapper<ProtoHelperNodeContext, IMLShapeInferenceContext, null_type> {
 public:
  MLKernelInferenceContext() = delete;
  MLKernelInferenceContext(
      OpNodeProtoHelper<ProtoHelperNodeContext>* info,  //todo - don't use node?
      const EdgeShapes* input_shapes_override,
      EdgeShapes& inferred_output_shapes) : OpNodeInfoWrapper(info, input_shapes_override), inferred_output_shapes_(inferred_output_shapes) {}

  ML_API_IMP(SetOutputTensorShape)(uint32_t output_index, uint32_t dimension_count, const int64_t* dimensions) noexcept override;

 private:
  EdgeShapes& inferred_output_shapes_;
};

class AbiCustomRegistry : public IMLOperatorRegistry {
 public:
  AbiCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry);

  ML_API_IMP(RegisterOpSetFromSchema)(
      const MLOperatorSetId* opSetId,
      int baseline_version,
      const MLSchemaDefinition* const* schema,
      uint32_t schema_count) const noexcept override;

  ML_API_IMP(RegisterOpKernel)(
      const MLOpKernelDefinition* op_kernel,
      MLOpKernelOptions options,
      IMLOpKernelCreateFn op_kernel_factory) const noexcept override;

  std::shared_ptr<CustomRegistry> GetRegistry() {
    return custom_registry_;
  }

 private:
  static OpSchema ConvertOpSchema(const char* domain, const MLSchemaDefinition& abi_schema);
  static std::string ConvertFormalParameterType(const MLFormalParameter& formal_parameter);
  static OpSchema::FormalParameterOption ConvertFormalParameterOption(MLFormalParameterOptions options);
  static void SetAttributesAndDefaults(OpSchema& schema, const MLSchemaDefinition& abi_schema);

  std::shared_ptr<CustomRegistry> custom_registry_;
};

}  // namespace Lotus

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

  template<class Base_t, class NodeInfoImpl_t>
  class OpNodeInfoWrapper : public Base_t {
  public:
    OpNodeInfoWrapper() = delete;
    OpNodeInfoWrapper(const OpNodeProtoHelper<NodeInfoImpl_t>* impl) : impl_(impl){}
    
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
    
    ML_API_IMP_(uint32_t, GetInputCount) () const noexcept override;
    ML_API_IMP_(uint32_t, GetOutputCount) () const noexcept override;

    ML_API_IMP(GetInputEdgeType) (uint32_t input_index, MLEdgeType* edge_type) const noexcept override;
    ML_API_IMP(GetOutputEdgeType) (uint32_t output_index, MLEdgeType* edge_type) const noexcept;

  protected:
    // Lifetime is managed by the caller and guaranteed to outlive this class
    const OpNodeProtoHelper<NodeInfoImpl_t>* impl_;

  private:
    template <MLAttributeType T>
    MLStatus GetAttributeHelper(
      const char* name,
      uint32_t element_byte_size,
      void* value) const;

    const std::string* GetStringAttribute(
      const char* name,
      uint32_t element_index) const;
  };

  class OpKernelInfoWrapper : public OpNodeInfoWrapper<IMLOpKernelInfo, ProtoHelperNodeContext>, public IMLOpKernelTensorShapeInfo {
  public:
    OpKernelInfoWrapper(const OpKernelInfo* kernel_info);

    // HasTensorShapeInfo returns false if and only if the kernel is registered using
    // MLOpKernelOptions::kAllowDynamicInputTensorSizes.  If this flag is specified and upstream
    // shapes are known when the kernel is created, HasTensorShapeInfo still returns false.
    ML_API_IMP_(bool, HasTensorShapeInfo) () const noexcept override;
    ML_API_IMP(GetTensorShapeInfo) (const IMLOpKernelTensorShapeInfo** shapeInfo) const noexcept override;

    ML_API_IMP_(const void*, GetExecutionHandle)() const noexcept override;
    
    // IMLOpKernelTensorShapeInfo methods
    ML_API_IMP(GetInputTensorDimensionCount)(uint32_t input_index, uint32_t* dimension_count) const noexcept override;
    ML_API_IMP(GetInputTensorShape)(uint32_t input_index, uint32_t dimension_count, int64_t* dimensions) const noexcept override;
    ML_API_IMP_(bool, HasOutputShapeInfo)() const noexcept override;
    ML_API_IMP(GetOutputTensorDimensionCount)(uint32_t output_index, uint32_t* dimension_count) const noexcept override;
    ML_API_IMP(GetOutputTensorShape)(uint32_t output_index, uint32_t dimension_count, int64_t* dimensions) const noexcept override;

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
    OpKernelContextWrapper(OpKernelContext* context, const IExecutionProvider* provider);

    ML_API_IMP(GetInputTensor)(uint32_t input_index, const IMLOpTensor** tensor) const noexcept override;
    ML_API_IMP(GetOutputTensor)(uint32_t output_index, IMLOpTensor** tensor) noexcept override;
    ML_API_IMP(GetDynamicOutputTensor)(uint32_t output_index, const int64_t* dimension_sizes, uint32_t dimensions, IMLOpTensor** tensor) noexcept override;

    ML_API_IMP(AllocateTemporaryData)(uint64_t size, void** data) const override;
    ML_API_IMP(FreeTemporaryData)(void* data) const override;

    ML_API_IMP_(const void*, GetExecutionHandle)() const noexcept override;

  protected:
    // Lifetime is managed by the caller and guaranteed to outlive this class
    OpKernelContext* impl_ = nullptr;

    // TODO - Use a custom STL allocator to avoid heap allocations in the common case.
    std::vector<TensorWrapper> inputTensors_;
    std::vector<TensorWrapper> outputTensors_;

    const IExecutionProvider* provider_ = nullptr;
  };

  class AbiOpKernel : public OpKernel {
  public:
    AbiOpKernel(IMLOpKernelCreateFn create_function, const OpKernelInfo& kernel_info);
    ~AbiOpKernel();

    Status Compute(OpKernelContext* context) const override;

  protected:
    // Lifetime is managed by the caller and guaranteed to outlive this class
    IMLOpKernel*  impl_ = nullptr;
  };

  class IShapeAndTypeInferenceContext : public IMLShapeInferenceContext, public IMLTypeInferenceContext {
  };

  class MLInferenceContext final : public OpNodeInfoWrapper<IShapeAndTypeInferenceContext, InferenceContext> {
    public:
      MLInferenceContext() = delete;
      MLInferenceContext(OpNodeProtoHelper<InferenceContext>* info, InferenceContext* ctx) : OpNodeInfoWrapper(info), context_(ctx){}

      ML_API_IMP(GetInputTensorDimensionCount)(uint32_t input_index, uint32_t* dimension_count) const noexcept override;
      ML_API_IMP(GetInputTensorShape)(uint32_t input_index, uint32_t dimension_count, int64_t* dimensions) const noexcept override;
      ML_API_IMP(SetOutputTensorShape)(uint32_t output_index, uint32_t dimension_count, const int64_t* dimensions) noexcept override;
      ML_API_IMP(SetOutputEdgeType)(uint32_t output_index, const MLEdgeType* edge_type) const noexcept override;

      InferenceContext* GetContext() const {
        return context_; 
      }

    private:
      InferenceContext* context_ = nullptr;
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
      static OpSchema ConvertOpSchema(const char* domain, const MLSchemaDefinition &abi_schema);
      static std::string ConvertFormalParameterType(const MLFormalParameter &formal_parameter);
      static OpSchema::FormalParameterOption ConvertFormalParameterOption(MLFormalParameterOptions options);
      static void SetAttributesAndDefaults(OpSchema& schema, const MLSchemaDefinition& abi_schema);
      static bool InputTensorShapesDefined(MLInferenceContext& abi_context);

      std::shared_ptr<CustomRegistry> custom_registry_;
  };

}  // namespace Lotus

//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#pragma once

#include "core/inc/op_kernel_author.h"

namespace Lotus {

typedef MLStatus (*IMLOpKernelCreateFn)(const IMLOpKernelInfo& kernelInfo, IMLOpKernel** opKernel);

class OpKernelInfoWrapper : public IMLOpKernelInfo {
 public:
  OpKernelInfoWrapper(const OpKernelInfo* kernelInfo);

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

  ML_API_IMP_(const void*, GetExecutionHandle)() const noexcept override;

 private:
  template <MLAttributeType T>
  MLStatus GetAttributeHelper(
      const char* name,
      uint32_t element_byte_size,
      void* value) const;

  const std::string* GetStringAttribute(
    const char* name,
    uint32_t element_index) const;

  // Lifetime is managed by the caller and guaranteed to outlive this class
  const OpKernelInfo* impl_ = nullptr;
};

class TensorWrapper : public IMLOpTensor {
 public:
  TensorWrapper() = default;

  TensorWrapper(Tensor* impl);

  ML_API_IMP(GetDimensionCount)(uint32_t* dimensions) const override;

  ML_API_IMP(GetDimensions)(
      int64_t* dimensions,
      uint32_t dimension_count) const noexcept override;

  ML_API_IMP_(MLTensorDataType, GetTensorDataType)() const noexcept override;

  ML_API_IMP_(MLBool, IsCPUData)() const noexcept override;

  ML_API_IMP_(MLBool, IsDataHandle)() const noexcept override;

  ML_API_IMP_(void*, GetData)() noexcept override;

  ML_API_IMP_(const void*, GetData)() const noexcept override;

  const Tensor* GetImpl() const { return nullptr; }
  Tensor* GetImpl() { return nullptr; }

 private:
  // Lifetime is managed by the caller and guaranteed to outlive this class
  Tensor* impl_ = nullptr;
};

class OpKernelContextWrapper : public IMLOpKernelContext {
 public:
  OpKernelContextWrapper(OpKernelContext* context);

  ML_API_IMP(GetInputEdgeType)(uint32_t input_index, MLEdgeType* edge_type) const noexcept override;

  ML_API_IMP(GetOutputEdgeType)(uint32_t output_index, MLEdgeType* edge_type) const noexcept override;

  ML_API_IMP(GetInputTensor)(uint32_t input_index, const IMLOpTensor** tensor) const noexcept override;

  ML_API_IMP(GetOutputTensor)(uint32_t output_index, const int64_t* dimension_sizes, uint32_t dimensions, IMLOpTensor** tensor) noexcept override;
  
  ML_API_IMP_(uint32_t, GetInputCount)() const noexcept override;
  ML_API_IMP_(uint32_t, GetOutputCount)() const noexcept override;

  ML_API_IMP(AllocateTemporaryData)(uint64_t size, void** data) const override;
  ML_API_IMP(FreeTemporaryData)(void* data) const override;

 protected:
  // Lifetime is managed by the caller and guaranteed to outlive this class
  OpKernelContext* impl_ = nullptr;

  // TODO - Use a custom STL allocator to avoid heap allocations in the common case.
  std::vector<TensorWrapper> inputTensors_;
  std::vector<TensorWrapper> outputTensors_;
};

class AbiOpKernel : public OpKernel {
 public:
   AbiOpKernel(IMLOpKernelCreateFn create_function, const OpKernelInfo& kernelInfo);
   ~AbiOpKernel();

  Status Compute(OpKernelContext* context) const override;

 protected:
  // Lifetime is managed by the caller and guaranteed to outlive this class
  IMLOpKernel* impl_ = nullptr;
};

}  // namespace Lotus
//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#pragma once

#include <cstdint>

typedef uint8_t MLBool;

// TODO: Merge status codes and exceptions with Lotus
enum class MLStatus : uint32_t {
  Success = 0,
  GenericFailure = 0xffffffff
};

// TODO - calling convention for former case
#if defined(__GNUC__)
#define ML_API(name) virtual MLStatus name
#define ML_API_(returnType, name) virtual returnType name
#else
#define ML_API(name) virtual MLStatus __stdcall name
#define ML_API_(returnType, name) virtual returnType __stdcall name
#endif

// Attribute types with numeric values matching the ONNX specification
enum class MLAttributeType {
  kUndefined = 0,
  kFloat = 2,
  kInt = 3,
  kString = 4,
  kFloatArray = 7,
  kIntArray = 8,
  kStringArray = 9
};

enum class MLTensorDataType {
  kUndefined = 0,
  kFloat = 1,
  kUInt8 = 2,
  kInt8 = 3,
  kUInt16 = 4,
  kInt16 = 5,
  kInt32 = 6,
  kInt64 = 7,
  kString = 8,
  kBool = 9,
  kFloat16 = 10,
  kDouble = 11,
  kUInt32 = 12,
  kUInt64 = 13,
  kComplex64 = 14,
  kComplex128 = 15
};

class IMLOpKernelInfo {
 public:
  // Gets the count of elements in an attribute
  ML_API(GetAttributeElementCount)(
      const char* name,
      uint32_t* elementCount) const noexcept = 0;
  
  // Returns whether an attribute with the specified name exists
  ML_API_(MLBool, HasAttribute)(const char* name) const noexcept = 0;

  // Gets the array of values in a numeric attribute
  ML_API(GetNumericAttribute)(
      const char* name,
      MLAttributeType type,
      uint32_t elementCount,
      uint32_t elementByteSize,
      void* value) const noexcept = 0;

  // Gets the size of an element within a UTF-8 string attribute,
  // including null termination
  ML_API(GetStringAttributeElementSize)(
      const char* name,
      uint32_t elementIndex,
      uint32_t* attributeElementSize) const noexcept = 0;

  // Gets the contents of an element within a UTF-8 string attribute.  The size
  // includes null termination.
  ML_API(GetStringAttributeElement)(
      const char* name,
      uint32_t elementIndex,
      uint32_t attributeElementSize,
      char* attributeElement) const noexcept = 0;
};

// Tensors methods used by implementations of IMLOpKernel::Compute
class IMLOpTensor {
 public:
  ML_API_(uint32_t, GetDimensionCount)() const noexcept = 0;

  ML_API(GetDimensions)(
      int64_t* dimensions,
      uint32_t dimensionCount) const noexcept = 0;

  ML_API_(MLTensorDataType, GetTensorDataType)() const noexcept = 0;

  ML_API_(MLBool, IsHostData)() const noexcept = 0;

  // This method should be called for tensors stored in host memory.
  // For outputs, SetDimensions should be called first.
  ML_API(GetHostData)(void** data) noexcept = 0;

  // This method should be called for tensors stored in host memory.
  // For outputs, SetDimensions should be called first.
  ML_API(GetHostData)(const void** data) const noexcept = 0;

  // Returns a handle whose type varies based on the kernel type.
  // This method should be called for tensors not stored in host memory.
  // For outputs, SetDimensions should be called first.
  // For D3D kernels this returns a pointer to an IUnknown supporting QueryInterface
  // to ID3D12Resource. D3D kernels should call Release on the returned value.
  ML_API(GetDataHandle)(void** data) const noexcept = 0;

  ML_API(SetDimensions)(
      const int64_t* dimensions,
      uint32_t dimensionCount) noexcept = 0;
};

enum class MLEdgeType {
  kUndefined = 0,
  kTensor = 1,
  kMap = 2
};

struct MLEdge {
  MLEdgeType type;

  union {
    IMLOpTensor* tensor;

    // (IMLMap, IMLSequence, etc.)
    // ...
  };

  inline bool IsTensor() {
    return type == MLEdgeType::kTensor;
  }
};

class IMLOpKernelContext {
 public:
  // Returns a handle whose type varies based on the kernel type.
  // For D3D kernels, this returns an IUnknown supporting QueryInterface to
  // ID3D12GraphicsCommandList1.   D3D kernels must call Release on the
  // returned value.
  ML_API_(void, GetExecutionHandle)(void** executionHandle) const noexcept;

  ML_API(GetInputEdge)(uint32_t inputIndex, MLEdge* edge) const noexcept;
  ML_API(GetOutputEdge)(uint32_t outputIndex, MLEdge* edge) const noexcept;
};

class IMLOpKernel {
 public:
  ML_API_(void, Release)() noexcept = 0;

  ML_API(Initialize)(const IMLOpKernelInfo* info) noexcept = 0;

  // Allocates and computes the outputs of the kernel.  The same IMLOpKernelInfo
  // is provided as to the Initialize method.  Tensors within the input and output
  // arrays have fully packed strides and have NCHW channel ordering.
  //
  // D3D kernels must assume each tensor is initially in the UAV state and should ensure
  // they are in the UAV state when returning.  Kernels must not depend on previous state set
  // within the command list.  The command list is executed on a compute queue,
  // and must contain only compute work.
  //
  // D3D kernels should cache pipeline state objects which they use within the command list
  // using ID3D12Object::SetPrivateDataInterface.
  ML_API(Compute)(const IMLOpKernelInfo* info, IMLOpKernelContext* context) noexcept = 0;
};
//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#pragma once

#include "operation_author.h"
#include <limits>

// TODO: Merge status codes and exceptions with Lotus
#define CHECK_STATUS(x)             \
  {                                 \
    if ((x) != MLStatus::Success) { \
      throw std::exception();       \
    }                               \
  }

#define CHECK_BOOL(x)         \
  {                           \
    if ((x) == false) {       \
      throw std::exception(); \
    }                         \
  }

//
// Traits for numeric attribute types
//
template <typename T>
struct MLAttributeTypeTraits {
};

template <>
struct MLAttributeTypeTraits<float> {
  const MLAttributeType AttributeType = MLAttributeType::kFloat;
  const MLAttributeType AttributeVectorType = MLAttributeType::kFloatArray;
};

template <>
struct MLAttributeTypeTraits<int> {
  const MLAttributeType AttributeType = MLAttributeType::kInt;
  const MLAttributeType AttributeVectorType = MLAttributeType::kIntArray;
};

//
// Traits for tensor types
//
template <typename T>
struct MLTensorTypeTraits {
};

template <>
struct MLTensorTypeTraits<float> {
  const MLTensorDataType TensorType = MLTensorDataType::kFloat;
};

template <>
struct MLTensorTypeTraits<uint8_t> {
  const MLTensorDataType TensorType = MLTensorDataType::kUInt8;
};

template <>
struct MLTensorTypeTraits<int8_t> {
  const MLTensorDataType TensorType = MLTensorDataType::kInt8;
};

template <>
struct MLTensorTypeTraits<uint16_t> {
  const MLTensorDataType TensorType = MLTensorDataType::kUInt16;
};

template <>
struct MLTensorTypeTraits<int16_t> {
  const MLTensorDataType TensorType = MLTensorDataType::kInt16;
};

template <>
struct MLTensorTypeTraits<int32_t> {
  const MLTensorDataType TensorType = MLTensorDataType::kInt32;
};

template <>
struct MLTensorTypeTraits<int64_t> {
  const MLTensorDataType TensorType = MLTensorDataType::kInt64;
};

template <>
struct MLTensorTypeTraits<double> {
  const MLTensorDataType TensorType = MLTensorDataType::kDouble;
};

template <>
struct MLTensorTypeTraits<uint32_t> {
  const MLTensorDataType TensorType = MLTensorDataType::kUInt32;
};

template <>
struct MLTensorTypeTraits<uint64_t> {
  const MLTensorDataType TensorType = MLTensorDataType::kUInt64;
};

//
// Wrappers for ABI objects consumed by kernels.
// These wrappers provide typesafe methods which use STL types and convert
// return values to exceptions.
//

class MLOpKernelInfo {
 public:
  MLOpKernelInfo(IMLOpKernelInfo* impl) : impl_(impl) {}

  uint32_t GetAttributeElementCount(const std::string& name) const {
    uint32_t elementCount;
    CHECK_STATUS(impl_->GetAttributeElementCount(name.c_str(), &elementCount));
    return elementCount;
  }

  bool HasAttribute(const std::string name) const noexcept {
    return impl_->HasAttribute(name.c_str());
  }

  //
  // Templatized methods to query numeric attributes using MLAttributeTypeTraits
  //
  template <typename T>
  T GetNumericAttribute(const std::string& name) const {
    T value;

    CHECK_STATUS(impl_->GetNumericAttribute(
        name.c_str(),
        MLAttributeTypeTraits<T>::AttributeType,
        1,
        sizeof(T),
        &value));

    return value;
  }

  template <typename T>
  std::vector<T> GetNumericAttributeVector(const std::string& name) const {
    uint32_t count = GetAttributeElementCount(name);
    std::vector<T> values(count);

    CHECK_STATUS(impl_->GetNumericAttribute(
        name.c_str(),
        MLAttributeTypeTraits<T>::AttributeVectorType,
        count,
        sizeof(T),
        values.data()));

    return std::move(values);
  }

  std::string GetStringAttribute(const std::string& name) const {
    return GetStringAttributeElement(name, 0);
  }

  std::vector<std::string> GetStringAttributeVector(const std::string& name) const {
    uint32_t count = GetAttributeElementCount(name);
    std::vector<std::string> values;
    values.resize(count);

    for (uint32_t i = 0; i < count; ++i) {
      values[i] = GetStringAttributeElement(name, i);
    }

    return std::move(values);
  }

  std::string GetStringAttributeElement(const std::string& name, uint32_t elementIndex) const {
    uint32_t size = 0;
    CHECK_STATUS(impl_->GetStringAttributeElementSize(name.c_str(), elementIndex, &size));

    // Construct a string by copying a character array.  The copy can be removed with C++17
    // using the non-const std::basic_string::data method.
    std::vector<char> temp(size);
    CHECK_STATUS(impl_->GetStringAttributeElement(name.c_str(), elementIndex, size, temp.data()));
    std::string value(temp.data());
    return std::move(value);
  }

 private:
  IMLOpKernelInfo* impl_;
};

class MLOpTensor {
 public:
  MLOpTensor(IMLOpTensor* impl) : impl_(impl) {}

  uint32_t GetDimensionCount() const {
    return impl_->GetDimensionCount();
  }

  std::vector<int64_t> GetDimensions() const {
    uint32_t dimensionCount = GetDimensionCount();
    std::vector<int64_t> dimensions(dimensionCount);
    CHECK_STATUS(impl_->GetDimensions(dimensions.data(), dimensionCount));
    return std::move(dimensions);
  }

  MLTensorDataType GetTensorDataType() const noexcept {
    return impl_->GetTensorDataType();
  }

  bool IsHostData() const noexcept {
    return impl_->IsHostData();
  }

  void GetHostData(void** data) {
    CHECK_BOOL(IsHostData());
    CHECK_STATUS(impl_->GetHostData(data));
  }

  void GetHostData(const void** data) const {
    CHECK_BOOL(IsHostData());
    CHECK_STATUS(impl_->GetHostData(data));
  }

  template <typename T>
  T* GetHostData() {
    CHECK_BOOL(GetTensorDataType() == MLTensorTypeTraits<T>::TensorType);

    T* data = nullptr;
    GetHostData(&data);
    return data;
  }

  template <typename T>
  const T* GetHostData() const {
    CHECK_BOOL(GetTensorDataType() == MLTensorTypeTraits<T>::TensorType);

    const T* data = nullptr;
    GetHostData(&data);
    return data;
  }

  void GetDataHandle(void** dataHandle) const {
    CHECK_BOOL(!IsHostData());
    CHECK_STATUS(impl_->GetDataHandle(dataHandle));
  }

  void SetDimensions(const std::vector<int64_t>& dimensions) {
    if (dimensions.size() > std::numeric_limits<uint32_t>::max()) {
      throw std::exception();
    }

    CHECK_STATUS(impl_->SetDimensions(
        dimensions.data(),
        static_cast<uint32_t>(dimensions.size())));
  }

 private:
  IMLOpTensor* impl_;
};

class MLOpKernelContext {
 public:
  MLOpKernelContext(IMLOpKernelContext* impl) : impl_(impl) {}

  void GetExecutionHandle(void** executionHandle) const {
    return impl_->GetExecutionHandle(executionHandle);
  }

  MLEdge GetInputEdge(uint32_t inputIndex) const {
    MLEdge edge;
    CHECK_STATUS(impl_->GetInputEdge(inputIndex, &edge));
    return edge;
  }

  MLEdge GetOutputEdge(uint32_t outputIndex) const {
    MLEdge edge;
    CHECK_STATUS(impl_->GetOutputEdge(outputIndex, &edge));
    return edge;
  }

  const MLOpTensor GetInputTensor(uint32_t inputIndex) const {
    MLEdge edge;
    CHECK_STATUS(impl_->GetInputEdge(inputIndex, &edge));

    CHECK_BOOL(edge.type == MLEdgeType::kTensor);
    return std::move(MLOpTensor(edge.tensor));
  }

  MLOpTensor GetOutputTensor(uint32_t outputIndex) const {
    MLEdge edge;
    CHECK_STATUS(impl_->GetOutputEdge(outputIndex, &edge));

    CHECK_BOOL(edge.type == MLEdgeType::kTensor);
    return std::move(MLOpTensor(edge.tensor));
  }

 private:
  IMLOpKernelContext* impl_;
};

// Helper class for operator implementations, templatized by the
// implementation type. This class converts ABI types to wrappers,
// supports STL / GSL types, and converts exceptions to return values.
template <class T>
class MLOpKernel : IMLOpKernel {
 public:
  MLOpKernel() {
  }

  virtual ~MLOpKernel() {
  }

  ML_API_(void, Release)() noexcept override {
    delete this;
  }

  ML_API(Initialize)(const IMLOpKernelInfo* info) noexcept override {
    try {
      _impl = std::make_unique<T>(MLOpKernelInfo(info));
      return MLStatus::Success;
    } catch (const std::exception& ex) {
      return MLStatus::GenericFailure;
    }
  }

  ML_API(Compute)(
      const IMLOpKernelInfo* info,
      IMLOpKernelContext* context) noexcept override {
    try {
      _impl->Compute(
          MLOpKernelInfo(info),
          MLOpKernelContext(context));

      return MLStatus::Success;
    } catch (const std::exception& ex) {
      return MLStatus::GenericFailure;
    }
  }

 protected:
  std::unique_ptr<T> _impl;
};
#pragma once

#include "core/common/status.h"
#include "core/graph/graph.h"
#include "gsl/span"

#ifdef __has_attribute
#define LOTUS_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define LOTUS_HAVE_ATTRIBUTE(x) 0
#endif

#if LOTUS_HAVE_ATTRIBUTE(nodiscard)
#define LOTUS_MUST_USE_RESULT [[nodiscard]]
#elif defined(__clang__) && LOTUS_HAVE_ATTRIBUTE(warn_unused_result)
#define LOTUS_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define LOTUS_MUST_USE_RESULT
#endif

class IMLOpKernel;

namespace onnxruntime {

// A set of wrappers with common signatures for use with both OpKernelInfo
// (as its base class) and InferenceContext.  Used by ABI kernels for both
// shape / type inference and kernel construction
template <class Impl_t>
class OpNodeProtoHelper {
 public:
  explicit OpNodeProtoHelper(const Impl_t* impl) : impl_(impl) {}

  //Get a single attribute
  template <typename T>
  LOTUS_MUST_USE_RESULT Status GetAttr(const std::string& name, T* value) const;

  //Get a single attribute
  template <typename T>
  T GetAttrOrDefault(const std::string& name, const T& default_value) const {
    T tmp;
    return GetAttr<T>(name, &tmp).IsOK() ? tmp : default_value;
  }

  //Get a single attribute
  template <typename T>
  void GetAttrOrDefault(const std::string& name, T* value, const T& default_value) const {
    if (!GetAttr<T>(name, value).IsOK())
      *value = default_value;
  }

  //Get repeated attributes
  template <typename T>
  LOTUS_MUST_USE_RESULT std::vector<T> GetAttrsOrDefault(const std::string& name, const std::vector<T>& default_value = std::vector<T>{}) const {
    std::vector<T> tmp;
    return GetAttrs<T>(name, tmp).IsOK() ? tmp : default_value;
  }

  //Get repeated attributes
  template <typename T>
  LOTUS_MUST_USE_RESULT Status GetAttrs(const std::string& name, std::vector<T>& values) const;

  template <typename T>
  LOTUS_MUST_USE_RESULT Status GetAttrs(const std::string& name, gsl::span<T> values) const;

  uint32_t GetPrimitiveAttrElementCount(onnx::AttributeProto_AttributeType type,
                                        const std::string& name) const noexcept;

  bool HasPrimitiveAttribute(onnx::AttributeProto_AttributeType type,
                             const std::string& name) const noexcept;

  uint32_t GetInputCount() const {
    return gsl::narrow_cast<uint32_t>(impl_->getNumInputs());
  }

  uint32_t GetOutputCount() const {
    return gsl::narrow_cast<uint32_t>(impl_->getNumOutputs());
  }

  const onnx::TypeProto* GetInputType(size_t index) const {
    return impl_->getInputType(index);
  }

  const onnx::TypeProto* GetOutputType(size_t index) const {
    // Work around lack of a const method from the onnx InferenceContext interface
    return const_cast<Impl_t*>(impl_)->getOutputType(index);
  }

  // Try to query an attribute, returning nullptr if it doesn't exist
  const onnx::AttributeProto* TryGetAttribute(const std::string& name) const {
    return impl_->getAttribute(name);
  }

  const onnx::AttributeProto* GetAttribute(const std::string& name) const {
    const onnx::AttributeProto* attr = TryGetAttribute(name);
    LOTUS_ENFORCE(attr != nullptr);
    return attr;
  }

 private:
  OpNodeProtoHelper() = delete;
  const Impl_t* impl_ = nullptr;
};

// The methods on the following class are called by OpNodeProtoHelper, implementing
// the same signatures as InferenceContext other than const-ness.
class ProtoHelperNodeContext {
 public:
  ProtoHelperNodeContext(const onnxruntime::Node& node) : node_(node) {}
  ProtoHelperNodeContext() = delete;

  const onnx::AttributeProto* getAttribute(const std::string& name) const;
  size_t getNumInputs() const;
  const onnx::TypeProto* getInputType(size_t index) const;
  size_t getNumOutputs() const;
  const onnx::TypeProto* getOutputType(size_t index) const;

 private:
  const onnxruntime::Node& node_;
};

}  // namespace onnxruntime

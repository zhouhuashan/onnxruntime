#pragma once

#include "core/common/status.h"
#include "core/graph/graph.h"
#include "gsl/span"

class IMLOpKernel;

namespace Lotus {

// A set of wrappers with common signatures for use with both OpKernelInfo
// (as its base class) and InferenceContext.  Used by ABI kernels for both
// shape / type inference and kernel construction
template<class Impl_t>
class OpNodeProtoHelper {
public:
  explicit OpNodeProtoHelper(const Impl_t *impl) : impl_(impl){}

  //Get a single attribute
  template <typename T>
  Status GetAttr(const std::string& name, T* value) const;

  //Get repeated attributes
  template <typename T>
  Status GetAttrs(const std::string& name, std::vector<T>& values) const;

  template <typename T>
  Status GetAttrs(const std::string& name, gsl::span<T> values) const;

  uint32_t GetPrimitiveAttrElementCount(AttributeProto_AttributeType type,
                                        const std::string& name) const noexcept;

  bool HasPrimitiveAttribute(AttributeProto_AttributeType type,
                             const std::string& name) const noexcept;

  uint32_t GetInputCount() const{
    return gsl::narrow_cast<uint32_t>(impl_->getNumInputs());
  }
          
  uint32_t GetOutputCount() const{
    return gsl::narrow_cast<uint32_t>(impl_->getNumOutputs());
  }
  
  const TypeProto* GetInputType(size_t index) const{
    return impl_->getInputType(index);
  }

  const TypeProto* GetOutputType(size_t index) const{
    // Work around lack of a const method from the onnx InferenceContext interface
    return const_cast<Impl_t *>(impl_)->getOutputType(index);
  }
  
  // Try to query an attribute, returning nullptr if it doesn't exist
  const AttributeProto* TryGetAttribute(const std::string& name) const {
    return impl_->getAttribute(name);  
  }

  const AttributeProto* GetAttribute(const std::string& name) const {
    const AttributeProto *attr = TryGetAttribute(name);
    LOTUS_ENFORCE(attr != nullptr);
    return attr;
  }

protected:
  OpNodeProtoHelper() = delete;
  const Impl_t* impl_ = nullptr;
};

// The methods on the following class are called by OpNodeProtoHelper, implementing 
// the same signatures as InferenceContext other than const-ness.
class ProtoHelperNodeContext {
public:
  ProtoHelperNodeContext(const LotusIR::Node& node) : node_(node){}
  ProtoHelperNodeContext() = delete;

  const AttributeProto* getAttribute(const std::string& name) const;
  size_t getNumInputs() const;
  const TypeProto* getInputType(size_t index) const;
  size_t getNumOutputs() const;
  const TypeProto* getOutputType(size_t index) const;

private:
  const LotusIR::Node& node_;
};

}  // namespace Lotus

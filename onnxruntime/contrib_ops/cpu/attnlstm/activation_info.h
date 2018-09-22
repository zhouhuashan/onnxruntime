#pragma once

namespace onnxruntime {
namespace ml {
namespace detail {

// Helper struct for an activation function call information
template <typename TFunc>
struct ActivationInfo {
  TFunc func;
  float alpha;
  float beta;
};

}  // namespace detail
}  // namespace ml
}  // namespace onnxruntime

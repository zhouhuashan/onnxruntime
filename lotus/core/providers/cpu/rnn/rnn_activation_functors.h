#ifndef RNN_ACTIVATION_FUNCTORS_H_
#define RNN_ACTIVATION_FUNCTORS_H_
#include <cmath>

#pragma warning(push)

#pragma warning(disable : 4100)

namespace Lotus {
namespace detail {

template <typename T>
inline T Affine(T x, T alpha, T beta) {
  return alpha * x + beta;
}

template <typename T>
inline T Relu(T x, T alpha = 0, T beta = 0) {
  return std::max(0.0f, static_cast<float>(x));
}

template <typename T>
inline T LeakyRelu(T x, T alpha, T beta = 0) {
  return x >= 0 ? x : alpha * x;
}

template <typename T>
inline T ThresholdedRelu(T x, T alpha, T beta = 0) {
  return x > alpha ? x : 0;
}

template <typename T>
inline T Sigmoid(T x, T alpha = 0, T beta = 0) {
  if (x >= 0) {
    return 1 / (1 + exp(-x));
  } else {
    return exp(x) / (1 + exp(x));
  }
}

template <typename T>
inline T Tanh(T x, T alpha = 0, T beta = 0) {
  return 2.0f * Sigmoid(2.0f * x) - 1.0f;
}

template <typename T>
inline T ScaledTanh(T x, T alpha, T beta) {
  return alpha * Tanh(beta * x);
}

template <typename T>
inline T HardSigmoid(T x, T alpha, T beta) {
  return std::min(1.0f, std::max(0.0f, alpha * x + beta));
}

template <typename T>
inline T Elu(T x, T alpha, T beta = 0) {
  return x >= 0 ? x : alpha * (exp(x) - 1);
}

template <typename T>
inline T Softsign(T x, T alpha = 0, T beta = 0) {
  return x / (1 + abs(x));
}

template <typename T>
inline T Softplus(T x, T alpha = 0, T beta = 0) {
  return log(1 + exp(x));
}

#pragma warning(pop)

template <typename T>
std::function<T(T, T, T)> GetFuncByName(const std::string& name, const std::string& default_name) {
  static std::unordered_map<std::string, std::function<T(T, T, T)>> NameToFuncMap(
      {{"Affine", Affine<T>},
       {"Relu", Relu<T>},
       {"LeakyRelu", LeakyRelu<T>},
       {"ThresholdedRelu", ThresholdedRelu<T>},
       {"Tanh", Tanh<T>},
       {"ScaledTanh", ScaledTanh<T>},
       {"Sigmoid", Sigmoid<T>},
       {"HardSigmoid", HardSigmoid<T>},
       {"Elu", Elu<T>},
       {"Softsign", Softsign<T>},
       {"Softplus", Softplus<T>}});

  if (NameToFuncMap.find(name) == NameToFuncMap.end()) {
    return NameToFuncMap[default_name];
  }
  return NameToFuncMap[name];
}

}  // namespace detail
}  // namespace Lotus

#endif  // RNN_ACTIVATION_FUNCTORS_H_

#include "core/providers/cpu/nn/pool.h"
#include <cmath>
using namespace Lotus::Common;

namespace Lotus {

class AveragePool {
 public:
  static float Initialize() {
    return 0.0;
  }

  template <typename T>
  static void Process(const T& x_data, T& y_data, const PoolProcessContext& /*cxt*/) {
    y_data += x_data;
  }

  template <typename T>
  static void Finalize(const int64_t size, T& y_data, const PoolProcessContext& /*cxt*/) {
    y_data /= size;
  }
};

class MaxPool {
 public:
  static float Initialize() {
    return std::numeric_limits<float>::lowest();
  }

  template <typename T>
  static void Process(const T& x_data, T& y_data, const PoolProcessContext& /*cxt*/) {
    if (x_data > y_data) {
      y_data = x_data;
    }
  }

  template <typename T>
  static void Finalize(const int64_t /*size*/, T& /*y_data*/, const PoolProcessContext& /*cxt*/) {}
};

class LpPool {
 public:
  static float Initialize() {
    return 0.0f;
  }

  template <typename T>
  static void Process(const T& x_data, T& y_data, const PoolProcessContext& cxt) {
    y_data += static_cast<T>(std::pow(std::abs(x_data), cxt.p_));
  }

  template <typename T>
  static void Finalize(const int64_t /*size*/, T& y_data, const PoolProcessContext& cxt) {
    y_data = static_cast<T>(std::pow(y_data, 1.0f / cxt.p_));
  }
};

template <typename T, typename PoolType>
Status Pool<T, PoolType>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  TensorShape x_shape = X->Shape();

  if (x_shape.NumDimensions() < 3) {
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "Input dimension cannot be less than 3.");
  }

  std::vector<int64_t> pads = pads_;
  std::vector<int64_t> kernel_shape = kernel_shape_;

  if (global_pooling_) {
    const auto& input_dims = x_shape.GetDims();
    kernel_shape.assign(input_dims.begin() + 2, input_dims.end());
    pads.assign(kernel_shape.size(), 0);
  }

  std::vector<int64_t> output_dims = PoolBase::SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, TensorShape(output_dims));

  const float* Xdata = X->template Data<float>();
  float* Ydata = Y->template MutableData<float>();
  // The main loop
  int64_t channels = x_shape[1];
  int64_t height = x_shape[2];
  int64_t width = kernel_shape.size() > 1 ? x_shape[3] : 1;
  int64_t depth = kernel_shape.size() > 2 ? x_shape[4] : 1;
  int64_t pooled_height = output_dims[2];
  int64_t pooled_width = kernel_shape.size() > 1 ? output_dims[3] : 1;
  int64_t pooled_depth = kernel_shape.size() > 2 ? output_dims[4] : 1;

  switch (kernel_shape.size()) {
    case 1:
      for (int64_t n = 0; n < x_shape[0]; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
          for (int64_t ph = 0; ph < pooled_height; ++ph) {
            int64_t hstart = ph * stride_h() - pads[0];
            int64_t hend = std::min(hstart + kernel_shape[0], height);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            T Yh = PoolType::Initialize();
            for (int64_t h = hstart; h < hend; ++h) {
              PoolType::Process(Xdata[h], Yh, pool_context_);
            }
            if (count_include_pad_) {
              PoolType::Finalize(kernel_shape[0], Yh, pool_context_);
            } else {
              PoolType::Finalize(hend - hstart, Yh, pool_context_);
            }
            Ydata[ph] = Yh;
          }
          // Do offset.
          Xdata += height;
          Ydata += pooled_height;
        }
      }
      break;
    case 2:
      for (int64_t n = 0; n < x_shape[0]; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
          for (int64_t ph = 0; ph < pooled_height; ++ph) {
            int64_t hstart = ph * stride_h() - pads[0];
            int64_t hend = std::min(hstart + kernel_shape[0], height);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            for (int64_t pw = 0; pw < pooled_width; ++pw) {
              int64_t wstart = pw * stride_w() - pads[1];
              int64_t wend = std::min(wstart + kernel_shape[1], width);
              wstart = std::max(wstart, static_cast<int64_t>(0));
              const int64_t pool_index = ph * pooled_width + pw;
              T Yh = PoolType::Initialize();
              for (int64_t h = hstart; h < hend; ++h) {
                for (int64_t w = wstart; w < wend; ++w) {
                  const int64_t input_index = h * width + w;
                  PoolType::Process(Xdata[input_index], Yh, pool_context_);
                }
              }
              if (count_include_pad_) {
                PoolType::Finalize(kernel_shape[0] * kernel_shape[1], Yh, pool_context_);
              } else {
                PoolType::Finalize((hend - hstart) * (wend - wstart), Yh, pool_context_);
              }
              Ydata[pool_index] = Yh;
            }
          }
          // Do offset.
          Xdata += height * width;
          Ydata += pooled_height * pooled_width;
        }
      }
      break;
    case 3:
      for (int64_t n = 0; n < x_shape[0]; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
          for (int64_t ph = 0; ph < pooled_height; ++ph) {
            int64_t hstart = ph * stride_h() - pads[0];
            int64_t hend = std::min(hstart + kernel_shape[0], height);
            hstart = std::max(hstart, static_cast<int64_t>(0));
            for (int64_t pw = 0; pw < pooled_width; ++pw) {
              int64_t wstart = pw * stride_w() - pads[1];
              int64_t wend = std::min(wstart + kernel_shape[1], width);
              wstart = std::max(wstart, static_cast<int64_t>(0));
              for (int64_t pd = 0; pd < pooled_depth; ++pd) {
                int64_t dstart = pd * stride_d() - pads[2];
                int64_t dend = std::min(dstart + kernel_shape[2], depth);
                dstart = std::max(dstart, static_cast<int64_t>(0));
                const int64_t pool_index =
                    ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
                T Yh = PoolType::Initialize();
                for (int64_t h = hstart; h < hend; ++h) {
                  for (int64_t w = wstart; w < wend; ++w) {
                    for (int64_t d = dstart; d < dend; ++d) {
                      const int64_t input_index = h * width * depth + w * depth + d;
                      PoolType::Process(Xdata[input_index], Yh, pool_context_);
                    }
                  }
                }
                if (count_include_pad_) {
                  PoolType::Finalize(kernel_shape[0] * kernel_shape[1] * kernel_shape[2], Yh, pool_context_);
                } else {
                  PoolType::Finalize(
                      (hend - hstart) * (wend - wstart) * (dend - dstart), Yh, pool_context_);
                }
                Ydata[pool_index] = Yh;
              }
            }
          }
          // Do offset.
          Xdata += height * width * depth;
          Ydata += pooled_height * pooled_width * pooled_depth;
        }
      }
      break;
    default:
      return Status(LOTUS, INVALID_ARGUMENT, "Unsupported pooling size : ");
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    AveragePool,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_KERNEL(
    MaxPool,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, MaxPool>);

ONNX_CPU_OPERATOR_KERNEL(
    LpPool,
    2,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, LpPool>);

ONNX_CPU_OPERATOR_KERNEL(
    GlobalLpPool,
    2,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, LpPool>);

ONNX_CPU_OPERATOR_KERNEL(
    GlobalAveragePool,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_KERNEL(
    GlobalMaxPool,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, MaxPool>);

}  // namespace Lotus

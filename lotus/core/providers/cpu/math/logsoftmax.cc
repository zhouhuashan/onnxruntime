#include "core/providers/cpu/math/logsoftmax.h"

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/softmax_shared.h"
#include "core/util/math.h"

namespace Lotus {

template <>
Status LogSoftmax<float>::Compute(OpKernelContext* ctx) const {
  const Tensor& X = *ctx->Input<Tensor>(0);
  const TensorShape input_shape{X.Shape()};

  Tensor* Y = ctx->Output(0, input_shape);

  size_t N = input_shape.SizeToDimension(axis_);
  size_t D = input_shape.SizeFromDimension(axis_);

  float* Ydata = Y->MutableData<float>();

  std::vector<float> scale_(N);
  std::vector<float> rowmax_(N);
  std::vector<float> sum_multiplier_(D, 1.f);  // initialize all multiplier values to 1.0

  const bool logarithmic = true;
  auto status = SoftmaxCPU(N, D, X.Data<float>(), Ydata,
                           scale_.data(), sum_multiplier_.data(), logarithmic, rowmax_.data());

  return status;
}

REGISTER_KERNEL(KernelDefBuilder("LogSoftmax")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                LogSoftmax<float>);

}  // namespace Lotus

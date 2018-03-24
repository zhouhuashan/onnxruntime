#include "core/providers/cpu/math/softmax.h"

#include "gsl/gsl_util"

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/softmax_shared.h"
#include "core/util/math.h"

namespace Lotus {

template <>
Status Softmax<float>::compute(OpKernelContext* ctx) const {
  const Tensor& X = *ctx->input<Tensor>(0);
  const TensorShape input_shape{X.shape()};

  Tensor* Y = ctx->output(0, input_shape);

  size_t N = input_shape.SizeToDimension(axis_);
  size_t D = input_shape.SizeFromDimension(axis_);

  float* Ydata = Y->mutable_data<float>();

  std::vector<float> scale_(N);
  std::vector<float> rowmax_(N);
  std::vector<float> sum_multiplier_(D, 1.f);  // initialize all multiplier values to 1.0

  const bool logarithmic = false;
  SoftmaxCPU(gsl::narrow_cast<int>(N), gsl::narrow_cast<int>(D), X.data<float>(), Ydata,
             scale_.data(), sum_multiplier_.data(), logarithmic, rowmax_.data());
  return Status::OK();
}

REGISTER_KERNEL(KernelDef("Softmax")
                    .Domain(LotusIR::c_onnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::c_cpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Softmax<float>);

}  // namespace Lotus

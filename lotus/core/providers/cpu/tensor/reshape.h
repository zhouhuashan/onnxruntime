#ifndef CORE_PROVIDERS_CPU_TENSOR_RESHAPE_H
#define CORE_PROVIDERS_CPU_TENSOR_RESHAPE_H

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "gsl/gsl_util"

namespace Lotus {

template <typename T>
class Reshape final : public OpKernel {
 public:
  Reshape(const OpKernelInfo& info) : OpKernel(info) {
    Status status = info.GetAttrs<int64_t>("shape", shape_);
    LOTUS_ENFORCE(status.IsOK(), "Attribute shape is not set.");
  }

  Status compute(OpKernelContext* context) const override {
    std::vector<int64_t> shape = shape_;
    int64_t unknown_dim = -1;
    const Tensor* X = context->input<Tensor>(0);
    const TensorShape& current_shape = X->shape();
    int64_t size = 1;

    for (int i = 0; i < shape.size(); ++i) {
      LOTUS_ENFORCE(shape[i] >= -1, "A dimension cannot be less than -1.");
      if (shape[i] == -1) {
        LOTUS_ENFORCE(unknown_dim == -1, "At most one dimension can be -1.");
        unknown_dim = i;
      } else if (shape[i] == 0) {
        LOTUS_ENFORCE(i < current_shape.NumDimensions(),
                      "The dimension with value zero exceeds"
                      " the dimension size of the input tensor.");
        shape[i] = current_shape[i];
        size *= current_shape[i];
      } else {
        size *= shape[i];
      }
    }

    if (unknown_dim != -1) {
      // calculate unknown dimension
      LOTUS_ENFORCE((current_shape.Size() % size) == 0,
                    "The input tensor cannot be reshaped to the requested shape");
      shape[unknown_dim] = current_shape.Size() / size;
    } else {
      // check if the output shape is valid.
      LOTUS_ENFORCE(gsl::narrow_cast<int64_t>(current_shape.Size()) == size,
                    "The input tensor cannot be reshaped to the requested shape");
    }

    Tensor* Y = context->output(0, TensorShape(shape));
    const std::vector<std::pair<int, int>>& alias = kernel_def().Alias();
    //If input X and output Y are not aliases, it means the kernel is not doing inplace operation.
    if (std::find(alias.begin(), alias.end(), std::pair<int, int>(0, 0)) == alias.end()) {
      //copying reshape
      for (int i = 0; i < current_shape.Size(); ++i) {
        Y->mutable_data<T>()[i] = X->data<T>()[i];
      }
    } else {  //non-copying reshape
      *(Y->mutable_data<T>()) = *(X->data<T>());
    }

    return Status::OK();
  }

 private:
  std::vector<int64_t> shape_;
};

}  //namespace Lotus

#endif  // !CORE_PROVIDERS_CPU_TENSOR_RESHAPE_H

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "gsl/gsl_util"

namespace Lotus {

template <typename T>
class Reshape final : public OpKernel {
 public:
  Reshape(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    // Copy the second input tensor into the shape vector
    const Tensor* shapeTensor = context->Input<Tensor>(1);
    LOTUS_ENFORCE(shapeTensor->Shape().NumDimensions() == 1,
                  "A shape tensor must be a vector tensor.");
    size_t nDims = static_cast<size_t>(shapeTensor->Shape()[0]);
    const int64_t* data = shapeTensor->Data<int64_t>();
    std::vector<int64_t> shape;
    for (size_t i = 0; i < nDims; ++i)
      shape.push_back(data[i]);

    int64_t unknown_dim = -1;
    const Tensor* X = context->Input<Tensor>(0);
    const TensorShape& X_shape = X->Shape();
    int64_t size = 1;
    for (size_t i = 0; i < nDims; ++i) {
      LOTUS_ENFORCE(shape[i] >= -1, "A dimension cannot be less than -1.");
      if (shape[i] == -1) {
        LOTUS_ENFORCE(unknown_dim == -1, "At most one dimension can be -1.");
        unknown_dim = i;
      } else {
        if (shape[i] == 0) {
          LOTUS_ENFORCE(i < X_shape.NumDimensions(),
                        "The dimension with value zero exceeds"
                        " the dimension size of the input tensor.");
          shape[i] = X_shape[i];
        }
        size *= shape[i];
      }
    }

    if (unknown_dim != -1) {
      // calculate unknown dimension
      LOTUS_ENFORCE((X_shape.Size() % size) == 0,
                    "The input tensor cannot be reshaped to the requested shape. Input shape:", X_shape,
                    " Output shape:", shape);
      shape[unknown_dim] = X_shape.Size() / size;
    } else {
      // check if the output shape is valid.
      LOTUS_ENFORCE(gsl::narrow_cast<int64_t>(X_shape.Size()) == size,
                    "The input tensor cannot be reshaped to the requested shape. Input shape:", X_shape,
                    " Output shape:", shape);
    }

    Tensor* Y = context->Output(0, TensorShape(shape));
    const T* source = X->Data<T>();
    T* target = Y->MutableData<T>();
    //If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      memcpy(target, source, X_shape.Size() * sizeof(T));
    }

    return Status::OK();
  }
};

template <typename T>
class Reshape_1 final : public OpKernel {
 public:
  Reshape_1(const OpKernelInfo& info) : OpKernel(info) {
    Status status = info.GetAttrs<int64_t>("shape", shape_);
    LOTUS_ENFORCE(status.IsOK(), "Attribute shape is not set.");
  }

  Status Compute(OpKernelContext* context) const override {
    std::vector<int64_t> shape = shape_;
    int64_t unknown_dim = -1;
    const Tensor* X = context->Input<Tensor>(0);
    const TensorShape& X_shape = X->Shape();
    int64_t size = 1;

    for (size_t i = 0; i < shape.size(); ++i) {
      LOTUS_ENFORCE(shape[i] >= -1, "A dimension cannot be less than -1.");
      if (shape[i] == -1) {
        LOTUS_ENFORCE(unknown_dim == -1, "At most one dimension can be -1.");
        unknown_dim = i;
      } else {
        if (shape[i] == 0) {
          LOTUS_ENFORCE(i < X_shape.NumDimensions(),
                        "The dimension with value zero exceeds"
                        " the dimension size of the input tensor.");
          shape[i] = X_shape[i];
        }
        size *= shape[i];
      }
    }

    if (unknown_dim != -1) {
      // calculate unknown dimension
      LOTUS_ENFORCE((X_shape.Size() % size) == 0,
                    "The input tensor cannot be reshaped to the requested shape");
      shape[unknown_dim] = X_shape.Size() / size;
    } else {
      // check if the output shape is valid.
      LOTUS_ENFORCE(gsl::narrow_cast<int64_t>(X_shape.Size()) == size,
                    "The input tensor cannot be reshaped to the requested shape");
    }

    Tensor* Y = context->Output(0, TensorShape(shape));
    const T* source = X->Data<T>();
    T* target = Y->MutableData<T>();
    //If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (target != source) {
      memcpy(target, source, X_shape.Size() * sizeof(T));
    }

    return Status::OK();
  }

 private:
  std::vector<int64_t> shape_;
};

}  //namespace Lotus

/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#pragma once

#include "core/providers/cpu/nn/conv_base.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace Lotus {
namespace {
inline void ComputeTransposePadAndOutputShape(
    const int64_t in_size,
    const int64_t stride,
    const int64_t kernel,
    const int64_t adj,
    AutoPadType pad_type,
    int64_t* pad_head,
    int64_t* pad_tail,
    int64_t* out_size) {
  if (*out_size != -1) {
    LOTUS_ENFORCE(*out_size >= 0);
    // total padding size
    int64_t paddings = std::max<int64_t>(0, (in_size - 1) * stride + kernel + adj - *out_size);
    if (pad_type == AutoPadType::SAME_UPPER) {  // pad more on head when paddings are odd.
      *pad_head = paddings - paddings / 2;
      *pad_tail = paddings / 2;
    } else {
      // for pad_type is NOTSET, SAME_LOWER or VALID
      // set pad_head as paddings/2, pad_tail as paddings-paddings/2.
      // That said, we pad more on tail when paddings are odd.
      *pad_head = paddings / 2;
      *pad_tail = paddings - paddings / 2;
    }
    return;
  } else {
    if (pad_type != AutoPadType::NOTSET) {
      switch (pad_type) {
          // We handle cases of AutoPadType::VALID and AutoPadType::SAME_UPPER/LOWER,
          // the same way
        case AutoPadType::VALID:
        case AutoPadType::SAME_UPPER:
        case AutoPadType::SAME_LOWER:
          *pad_head = 0;
          *pad_tail = 0;
          *out_size = (in_size - 1) * stride + kernel + adj;
          break;
        default:
          throw NotImplementedException("pad type not supported");
      }
    } else {
      *out_size =
          (in_size - 1) * stride + kernel + adj - *pad_head - *pad_tail;
    }
  }
}
}  // namespace
template <typename T>
class ConvTranspose : public OpKernel, public ConvBase {
 public:
  ConvTranspose(const OpKernelInfo& info) : OpKernel(info), ConvBase(info), output_shape_(info.GetAttrsOrDefault<int64_t>("output_shape")), output_padding_(info.GetAttrsOrDefault<int64_t>("output_padding")) {
  }

  Status Compute(OpKernelContext* context) const override {
    size_t num_inputs = OpKernel::Node().InputDefs().size();
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* F = context->Input<Tensor>(1);
    const Tensor* B = num_inputs == 3 ? context->Input<Tensor>(2) : nullptr;
    const TensorShape& input_shape = X->Shape();

    // input validations
    if (group_ <= 0) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "group count is <= 0",
                               " group: ", group_);
    }

    if (input_shape.NumDimensions() != 4) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Input X must be 4-dimensional.",
                               " X: ", X->Shape().ToString().c_str());
    }

    if (input_shape.NumDimensions() != F->Shape().NumDimensions()) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "X num_dims does not match W num_dims.",
                               " X: ", X->Shape().ToString().c_str(),
                               " W: ", F->Shape().ToString().c_str());
    }

    const int64_t num_input_channels = input_shape[1];

    if (F->Shape()[0] != num_input_channels) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "filter number not equal to input channel number.",
                               " filter_number: ", F->Shape()[0],
                               " num_input_channels: ", num_input_channels);
    }

    const int64_t N = input_shape[0];
    const int64_t H = input_shape[2];
    const int64_t W = input_shape[3];
    const int64_t num_output_channels_multiplier = F->Shape()[1];
    const int64_t num_output_channels = num_output_channels_multiplier * group_;

    // it looks like num_output_channels is really k*group_ similar to how in the conv case
    // num_input_channels is k*group_. hence removing the check for num_output_channels here.

    if (num_input_channels % group_ != 0) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Input channels is not divisible by group.",
                               " num_input_channels: ", num_input_channels,
                               " group: ", group_);
    }

    std::vector<int64_t> kernel_shape = ComputeKernelShape(F->Shape());

    if (kernel_shape[0] != F->Shape()[2]) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "kernel height does not match filter height.",
                               " kernel_height: ", kernel_shape[0],
                               " filter_height: ", F->Shape()[2]);
    }

    if (kernel_shape[1] != F->Shape()[3]) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "kernel width does not match filter width.",
                               " kernel_width: ", kernel_shape[1],
                               " filter_width: ", F->Shape()[3]);
    }

    std::vector<int64_t> output_padding(output_padding_);
    if (output_padding.empty()) {
      output_padding.resize(kernel_shape.size(), 0);
    }
    std::vector<int64_t> pads(pads_);
    if (pads.empty()) {
      pads.resize(kernel_shape.size() * 2, 0);
    }
    std::vector<int64_t> dilations(dilations_);
    if (dilations.empty()) {
      dilations.resize(kernel_shape.size(), 1);
    }
    std::vector<int64_t> strides(strides_);
    if (strides.empty()) {
      strides.resize(kernel_shape.size(), 1);
    }

    std::vector<int64_t> Y_dims;

    ComputePadsAndOutputShape(input_shape, num_output_channels, kernel_shape, strides, output_padding, &pads, &Y_dims);
    TensorShape Yshape(Y_dims);
    Tensor* Y = context->Output(0, Yshape);

    const int64_t input_image_size = H * W;
    const int64_t X_offset = num_input_channels / group_ * input_image_size;
    const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / group_;
    const int64_t W_offset = F->Shape().Size() / group_;
    const int64_t kernel_dim = num_output_channels / group_ * kernel_shape[0] * kernel_shape[1];
    const int64_t output_image_size = Y_dims[2] * Y_dims[3];

    AllocatorPtr alloc;
    LOTUS_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

    auto col_data = alloc->Alloc(sizeof(T) * kernel_dim * H * W);
    BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
    T* col_buffer_data = static_cast<T*>(col_buffer.get());

    const T* Xdata = X->template Data<T>();
    const T* filter_data = F->template Data<T>();
    T* Ydata = Y->template MutableData<T>();

    for (auto image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        // Weight term
        Math::Gemm<T, CPUMathUtil>(
            CblasTrans,
            CblasNoTrans,
            kernel_dim,
            input_image_size,
            num_input_channels / group_,
            1,
            filter_data + group_id * W_offset,
            Xdata + group_id * X_offset,
            0,
            col_buffer_data,
            &CPUMathUtil::Instance());

        // Col2im
        Math::Col2im<T, CPUMathUtil, StorageOrder::NCHW>(
            col_buffer_data,
            num_output_channels / group_,
            Y_dims[2],
            Y_dims[3],
            kernel_shape[0],
            kernel_shape[1],
            1,
            1,
            pads[0],
            pads[1],
            pads[2],
            pads[3],
            strides[0],
            strides[1],
            Ydata + group_id * Y_offset,
            &CPUMathUtil::Instance());
      }

      if (B != nullptr) {
        auto Ymatrix = EigenMatrixMap<T>(Ydata, output_image_size, num_output_channels);
        auto Bvec = ConstEigenVectorMap<T>(B->template Data<T>(), num_output_channels);
        Ymatrix.rowwise() += Bvec.transpose();
      }

      Xdata += X_offset * group_;
      Ydata += Y_offset * group_;
    }

    return Status::OK();
  }

 private:
  const std::vector<int64_t> output_padding_;
  const std::vector<int64_t> output_shape_;

  void ComputePadsAndOutputShape(const TensorShape input_shape,
                                 const int64_t output_channel,
                                 const std::vector<int64_t>& kernel_shape,
                                 const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& output_padding,
                                 std::vector<int64_t>* pads,
                                 std::vector<int64_t>* output_shape) const {
    const int64_t N = input_shape[0];
    const int64_t H = input_shape[2];
    const int64_t W = input_shape[3];
    int64_t output_height = -1, output_width = -1;
    size_t output_shape_size = output_shape_.size();

    if (output_shape_size != 0) {
      output_height = output_shape_[output_shape_size - 2];
      output_width = output_shape_[output_shape_size - 1];
      LOTUS_ENFORCE(output_height >= H, "Output height cannot be smaller than input height.");
      LOTUS_ENFORCE(output_width >= W, "Output width cannot be smaller than input width.");
    }

    ComputeTransposePadAndOutputShape(
        H,
        strides[0],
        kernel_shape[0],
        output_padding[0],
        auto_pad_,
        &pads->at(0),
        &pads->at(2),
        &output_height);

    ComputeTransposePadAndOutputShape(
        W,
        strides[1],
        kernel_shape[1],
        output_padding[1],
        auto_pad_,
        &pads->at(1),
        &pads->at(3),
        &output_width);

    output_shape->insert(output_shape->begin(), {N, output_channel, output_height, output_width});
  }
};

}  // namespace Lotus

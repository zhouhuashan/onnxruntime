#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "gsl/gsl_util"

namespace Lotus {

template <typename T>
class Crop final : public OpKernel {
 public:
  Crop(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrs("border", border_);
    info.GetAttrs("scale", scale_);
  }

  Common::Status Compute(OpKernelContext* context) const override {
    if (border_.size() < 4) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT,
                               "Attribute border needs to be specified with four border elements, got ", border_.size());
    }

    const Tensor* X = context->Input<Tensor>(0);
    const auto dims = X->Shape().GetDims();

    if (dims.size() < 4) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT,
                               "Input is expected to have four dimensions corresponding to [N,C,H,W], got ", dims.size());
    }

    const int64_t N = dims[0];
    const int64_t C = dims[1];
    const int64_t H = dims[2];
    const int64_t W = dims[3];

    // find the cropped region, and copy it to the destination matrix
    int64_t leftBorder = border_[0],
            topBorder = border_[1],
            rightBorder = border_[2],
            bottomBorder = border_[3];

    if (H < topBorder + bottomBorder) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Input's height (", H, ") needs to be greater than the topBorder (", topBorder, ") + bottomBorder (", bottomBorder, ")");
    }

    if (W < leftBorder + rightBorder) {
      return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Input's width (", W, ") needs to be greater than the leftBorder (", leftBorder, ") + rightBorder (", rightBorder, ")");
    }

    int64_t bottomLimit = H - bottomBorder;
    int64_t rightLimit = W - rightBorder;

    // scale = (height, width)
    if (!scale_.empty()) {
      bottomLimit = topBorder + scale_[0];
      rightLimit = leftBorder + scale_[1];

      if (H < bottomLimit) {
        return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT,
                                 "Input's height (", H, ") needs to be greater than the topBorder (", topBorder, ") + scale_[0] (", scale_[0], ")");
      }

      if (W < rightLimit) {
        return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "Input's width (", W, ") needs to be greater than the leftBorder (", leftBorder, ") + scale_[1] (", scale_[1], ")");
      }
    }

    Tensor* Y = context->Output(0, TensorShape({N, C, bottomLimit - topBorder, rightLimit - leftBorder}));
    const T* Xdata = X->Data<T>();
    T* Ydata = Y->MutableData<T>();

    int64_t dest_idx = 0;
    int64_t HW = H * W;
    int64_t CHW = C * HW;
    int64_t nCHW;
    int64_t nCHW_p_cHW;
    int64_t nCHW_p_cHW_p_hW;
    int64_t source_idx;
    for (int64_t n = 0; n < N; ++n) {
      nCHW = n * CHW;
      for (int64_t c = 0; c < C; ++c) {
        nCHW_p_cHW = nCHW + c * HW;
        for (int64_t h = topBorder; h < bottomLimit; ++h) {
          nCHW_p_cHW_p_hW = nCHW_p_cHW + h * W;
          for (int64_t w = leftBorder; w < rightLimit; ++w) {
            source_idx = nCHW_p_cHW_p_hW + w;
            Ydata[dest_idx++] = Xdata[source_idx];
          }
        }
      }
    }
    return Status::OK();
  }

 private:
  std::vector<int64_t> border_;  // (leftBorder, topBorder, rightBorder, bottomBorder)
  std::vector<int64_t> scale_;   // (height, width)
};

}  //namespace Lotus

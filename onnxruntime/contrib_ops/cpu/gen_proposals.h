// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "gen_proposals_eigen_utils.h"
#include <cctype>

namespace onnxruntime {
namespace contrib {

namespace utils {

// A sub tensor view
template <class T>
class ConstTensorView {
 public:
  ConstTensorView(const T* data, const std::vector<int64_t>& dims)
      : data_(data), dims_(dims) {}

  size_t ndim() const {
    return dims_.size();
  }
  const std::vector<int64_t>& dims() const {
    return dims_;
  }
  int64_t dim(int i) const {
    return dims_.at(i);
  }
  const T* data() const {
    return data_;
  }
  size_t size() const {
    size_t size_ret = 1;
    for (auto& elem : dims_) {
      size_ret *= elem;
    }
    return size_ret;
  }

 private:
  const T* data_ = nullptr;
  std::vector<int64_t> dims_;
};
}  // namespace utils

template <typename T>
class GenerateProposals final : public OpKernel {
 public:
  explicit GenerateProposals(const OpKernelInfo& info) : OpKernel(info) {
    // min size
    float min_size_tmp;
    if (info.GetAttr<float>("min_size", &min_size_tmp).IsOK()) {
      min_size_ = min_size_tmp;
    }

    // nms_thresh
    float nms_thresh_tmp;
    if (info.GetAttr<float>("nms_thresh", &nms_thresh_tmp).IsOK()) {
      nms_thresh_ = nms_thresh_tmp;
    }

    // post_nms_topN
    int64_t post_nms_topN_tmp;
    if (info.GetAttr<int64_t>("post_nms_topN", &post_nms_topN_tmp).IsOK()) {
      post_nms_topN_ = post_nms_topN_tmp;
    }

    // pre_nms_topN
    int64_t pre_nms_topN_tmp;
    if (info.GetAttr<int64_t>("pre_nms_topN", &pre_nms_topN_tmp).IsOK()) {
      pre_nms_topN_ = pre_nms_topN_tmp;
    }

    // spatial_scale
    float spatial_scale_tmp;
    if (info.GetAttr<float>("spatial_scale", &spatial_scale_tmp).IsOK()) {
      spatial_scale_ = spatial_scale_tmp;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  void ProposalsForOneImage(
      const Eigen::Array3f& im_info,
      const Eigen::Map<const ERArrXXf>& anchors,
      const utils::ConstTensorView<float>& bbox_deltas_tensor,
      const utils::ConstTensorView<float>& scores_tensor,
      ERArrXXf* out_boxes,
      EArrXf* out_probs) const;

  // spatial_scale_ must be declared before feat_stride_
  float spatial_scale_{1.0};
  float feat_stride_{1.0};

  // RPN_PRE_NMS_TOP_N
  int64_t pre_nms_topN_{6000};
  // RPN_POST_NMS_TOP_N
  int64_t post_nms_topN_{300};
  // RPN_NMS_THRESH
  float nms_thresh_{0.7f};
  // RPN_MIN_SIZE
  float min_size_{16};
  // Correct bounding box transform coordates, see bbox_transform() in boxes.py
  // Set to true to match the detectron code, set to false for backward
  // compatibility
  bool correct_transform_coords_{false};
  // If set, for rotated boxes in RRPN, output angles are normalized to be
  // within [angle_bound_lo, angle_bound_hi].
  bool angle_bound_on_{true};
  int angle_bound_lo_{-90};
  int angle_bound_hi_{90};
  // For RRPN, clip almost horizontal boxes within this threshold of
  // tolerance for backward compatibility. Set to negative value for
  // no clipping.
  float clip_angle_thresh_{1.0};

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GenerateProposals);
};
}  // namespace contrib
}  // namespace onnxruntime

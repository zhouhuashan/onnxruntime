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

#include "gen_proposals.h"
#include "gen_proposals_op_util_nms.h"
#include "gen_proposals_op_util_boxes.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

#define ADD_TYPED_GENERATE_PROPOSALS_OP(data_type)                        \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                      \
      GenerateProposals,                                                  \
      1,                                                                  \
      data_type,                                                          \
      KernelDefBuilder()                                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      GenerateProposals<data_type>);

ADD_TYPED_GENERATE_PROPOSALS_OP(float);
ADD_TYPED_GENERATE_PROPOSALS_OP(double);

namespace utils {

// Generate a list of bounding box shapes for each pixel based on predefined
//     bounding box shapes 'anchors'.
// anchors: predefined anchors, size(A, 4)
// Return: all_anchors_vec: (H * W, A * 4)
// Need to reshape to (H * W * A, 4) to match the format in python
ERMatXf ComputeAllAnchors(
    const Tensor& anchors,
    int height,
    int width,
    float feat_stride) {
  const auto K = height * width;
  const auto A = anchors.Shape().GetDims()[0];
  const auto box_dim = anchors.Shape().GetDims()[1];  // TODO don't call Shape().GetDims() again
  ORT_ENFORCE(box_dim == 4 || box_dim == 5);

  ERMatXf shift_x = (ERVecXf::LinSpaced(width, 0.0f, width - 1.0f) * feat_stride)
                        .replicate(height, 1);
  ERMatXf shift_y = (EVecXf::LinSpaced(height, 0.0f, height - 1.0f) * feat_stride)
                        .replicate(1, width);
  Eigen::MatrixXf shifts(K, box_dim);
  if (box_dim == 4) {
    // Upright boxes in [x1, y1, x2, y2] format
    shifts << ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
        ConstEigenVectorMap<float>(shift_y.data(), shift_y.size()),
        ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
        ConstEigenVectorMap<float>(shift_y.data(), shift_y.size());
  } else {
    // Rotated boxes in [ctr_x, ctr_y, w, h, angle] format.
    // Zero shift for width, height and angle.
    ERMatXf shift_zero = ERMatXf::Constant(height, width, 0.0);
    shifts << ConstEigenVectorMap<float>(shift_x.data(), shift_x.size()),
        ConstEigenVectorMap<float>(shift_y.data(), shift_y.size()),
        ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size()),
        ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size()),
        ConstEigenVectorMap<float>(shift_zero.data(), shift_zero.size());
  }

  // Broacast anchors over shifts to enumerate all anchors at all positions
  // in the (H, W) grid:
  //   - add A anchors of shape (1, A, box_dim) to
  //   - K shifts of shape (K, 1, box_dim) to get
  //   - all shifted anchors of shape (K, A, box_dim)
  //   - reshape to (K*A, box_dim) shifted anchors
  ConstEigenMatrixMap<float> anchors_vec(
      anchors.template Data<float>(), 1, A * box_dim);
  // equivalent to python code
  //  all_anchors = (
  //        self._model.anchors.reshape((1, A, box_dim)) +
  //        shifts.reshape((1, K, box_dim)).transpose((1, 0, 2)))
  //    all_anchors = all_anchors.reshape((K * A, box_dim))
  // all_anchors_vec: (K, A * box_dim)
  ERMatXf all_anchors_vec =
      anchors_vec.replicate(K, 1) + shifts.rowwise().replicate(A);

  // use the following to reshape to (K * A, box_dim)
  // Eigen::Map<const ERMatXf> all_anchors(
  //            all_anchors_vec.data(), K * A, box_dim);

  return all_anchors_vec;
}

// Like ComputeAllAnchors, but instead of computing anchors for every single
// spatial location, only computes anchors for the already sorted and filtered
// positions after NMS is applied to avoid unnecessary computation.
// `order` is a raveled array of sorted indices in (A, H, W) format.
ERArrXXf ComputeSortedAnchors(
    const Eigen::Map<const ERArrXXf>& anchors,
    int height,
    int width,
    float feat_stride,
    const std::vector<int>& order) {
  const auto box_dim = anchors.cols();
  ORT_ENFORCE(box_dim == 4 || box_dim == 5);

  // Order is flattened in (A, H, W) format. Unravel the indices.
  const auto& order_AHW = utils::AsEArrXt(order);
  const auto& order_AH = order_AHW / width;
  const auto& order_W = order_AHW - order_AH * width;
  const auto& order_A = order_AH / height;
  const auto& order_H = order_AH - order_A * height;

  // Generate shifts for each location in the H * W grid corresponding
  // to the sorted scores in (A, H, W) order.
  const auto& shift_x = order_W.cast<float>() * feat_stride;
  const auto& shift_y = order_H.cast<float>() * feat_stride;
  Eigen::MatrixXf shifts(order.size(), box_dim);
  if (box_dim == 4) {
    // Upright boxes in [x1, y1, x2, y2] format
    shifts << shift_x, shift_y, shift_x, shift_y;
  } else {
    // Rotated boxes in [ctr_x, ctr_y, w, h, angle] format.
    // Zero shift for width, height and angle.
    const auto& shift_zero = EArrXf::Constant(order.size(), 0.0);
    shifts << shift_x, shift_y, shift_zero, shift_zero, shift_zero;
  }

  // Apply shifts to the relevant anchors.
  // Equivalent to python code `all_anchors = self._anchors[order_A] + shifts`
  ERArrXXf anchors_sorted;
  utils::GetSubArrayRows(anchors, order_A, &anchors_sorted);
  const auto& all_anchors_sorted = anchors_sorted + shifts.array();
  return all_anchors_sorted;
}

}  // namespace utils

// Compute the 1-d index of a n-dimensional contiguous row-major tensor for
//     a given n-dimensional index 'index'
size_t ComputeStartIndex(
    const Tensor& tensor,
    const std::vector<int>& index) {
  ORT_ENFORCE_EQ(index.size(), tensor.Shape().NumDimensions());

  size_t ret = 0;
  for (int i = 0; i < index.size(); i++) {
    ret += index[i] * tensor.Shape().SizeFromDimension(i + 1);
  }

  return ret;
}

// Get a sub tensor view from 'tensor' using data pointer from 'tensor'
template <class T>
utils::ConstTensorView<T> GetSubTensorView(
    const Tensor& tensor,
    int dim0_start_index) {
  //ORT_ENFORCE_EQ(tensor.dtype().itemsize(), sizeof(T)); // TODO

  if (tensor.Shape().Size() == 0) {
    return utils::ConstTensorView<T>(nullptr, {});
  }

  std::vector<int> start_dims(tensor.Shape().NumDimensions(), 0);
  start_dims.at(0) = dim0_start_index;
  auto st_idx = ComputeStartIndex(tensor, start_dims);
  auto ptr = tensor.Data<T>() + st_idx;

  auto input_dims = tensor.Shape().GetDims();
  std::vector<int> ret_dims(input_dims.begin() + 1, input_dims.end());

  utils::ConstTensorView<T> ret(ptr, ret_dims);
  return ret;
}

template <typename T>
void GenerateProposals<T>::ProposalsForOneImage(
    const Eigen::Array3f& im_info,
    const Eigen::Map<const ERArrXXf>& anchors,
    const utils::ConstTensorView<float>& bbox_deltas_tensor,
    const utils::ConstTensorView<float>& scores_tensor,
    ERArrXXf* out_boxes,
    EArrXf* out_probs) const {
  const auto& post_nms_topN = post_nms_topN_;
  const auto& nms_thresh = nms_thresh_;
  const auto& min_size = min_size_;
  const int box_dim = static_cast<int>(anchors.cols());
  ORT_ENFORCE(box_dim == 4 || box_dim == 5);

  ORT_ENFORCE_EQ(bbox_deltas_tensor.ndim(), 3);
  ORT_ENFORCE_EQ(bbox_deltas_tensor.dim(0) % box_dim, 0);
  auto A = bbox_deltas_tensor.dim(0) / box_dim;
  auto H = bbox_deltas_tensor.dim(1);
  auto W = bbox_deltas_tensor.dim(2);
  auto K = H * W;
  ORT_ENFORCE_EQ(A, anchors.rows());

  // scores are (A, H, W) format from conv output.
  // Maintain the same order without transposing (which is slow)
  // and compute anchors accordingly.
  ORT_ENFORCE_EQ(scores_tensor.ndim(), 3);
  ORT_ENFORCE_EQ(scores_tensor.dims(), (std::vector<int>{A, H, W}));
  Eigen::Map<const EArrXf> scores(scores_tensor.data(), scores_tensor.size());

  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  if (pre_nms_topN_ <= 0 || pre_nms_topN_ >= scores.size()) {
    // 4. sort all (proposal, score) pairs by score from highest to lowest
    // 5. take top pre_nms_topN (e.g. 6000)
    std::sort(order.begin(), order.end(), [&scores](int lhs, int rhs) {
      return scores[lhs] > scores[rhs];
    });
  } else {
    // Avoid sorting possibly large arrays; First partition to get top K
    // unsorted and then sort just those (~20x faster for 200k scores)
    std::partial_sort(
        order.begin(),
        order.begin() + pre_nms_topN_,
        order.end(),
        [&scores](int lhs, int rhs) { return scores[lhs] > scores[rhs]; });
    order.resize(pre_nms_topN_);
  }

  EArrXf scores_sorted;
  utils::GetSubArray(scores, utils::AsEArrXt(order), &scores_sorted);

  // bbox_deltas are (A * box_dim, H, W) format from conv output.
  // Order them based on scores maintaining the same format without
  // expensive transpose.
  // Note that order corresponds to (A, H * W) in row-major whereas
  // bbox_deltas are in (A, box_dim, H * W) in row-major. Hence, we
  // obtain a sub-view of bbox_deltas for each dim (4 for RPN, 5 for RRPN)
  // in (A, H * W) with an outer stride of box_dim * H * W. Then we apply
  // the ordering and filtering for each dim iteratively.
  ERArrXXf bbox_deltas_sorted(order.size(), box_dim);
  EArrXf bbox_deltas_per_dim(A * K);
  EigenOuterStride stride(box_dim * K);
  for (int j = 0; j < box_dim; ++j) {
    Eigen::Map<ERMatXf>(bbox_deltas_per_dim.data(), A, K) =
        Eigen::Map<const ERMatXf, 0, EigenOuterStride>(
            bbox_deltas_tensor.data() + j * K, A, K, stride);
    for (int i = 0; i < order.size(); ++i) {
      bbox_deltas_sorted(i, j) = bbox_deltas_per_dim[order[i]];
    }
  }

  // Compute anchors specific to the ordered and pre-filtered indices
  // in (A, H, W) format.
  const auto& all_anchors_sorted =
      utils::ComputeSortedAnchors(anchors, H, W, feat_stride_, order);

  // Transform anchors into proposals via bbox transformations
  static const std::vector<float> bbox_weights{1.0, 1.0, 1.0, 1.0};
  auto proposals = utils::bbox_transform(
      all_anchors_sorted,
      bbox_deltas_sorted,
      bbox_weights,
      utils::BBOX_XFORM_CLIP_DEFAULT,
      correct_transform_coords_,
      angle_bound_on_,
      angle_bound_lo_,
      angle_bound_hi_);

  // 2. clip proposals to image (may result in proposals with zero area
  // that will be removed in the next step)
  proposals =
      utils::clip_boxes(proposals, static_cast<int>(im_info[0]), static_cast<int>(im_info[1]), clip_angle_thresh_);

  // 3. remove predicted boxes with either height or width < min_size
  auto keep = utils::filter_boxes(proposals, min_size, im_info);
  ORT_ENFORCE_LE(keep.size(), static_cast<size_t>(scores_sorted.size()));

  // 6. apply loose nms (e.g. threshold = 0.7)
  // 7. take after_nms_topN (e.g. 300)
  // 8. return the top proposals (-> RoIs top)
  if (post_nms_topN > 0 && static_cast<size_t>(post_nms_topN) < keep.size()) {
    keep = utils::nms_cpu(
        proposals, scores_sorted, keep, nms_thresh, post_nms_topN);
  } else {
  }

  // Generate outputs
  utils::GetSubArrayRows(proposals, utils::AsEArrXt(keep), out_boxes);
  utils::GetSubArray(scores_sorted, utils::AsEArrXt(keep), out_probs);
}

template <typename T>
Status GenerateProposals<T>::Compute(OpKernelContext* ctx) const {
  const auto& scores = *ctx->Input<Tensor>(0);
  const auto& bbox_deltas = *ctx->Input<Tensor>(1);
  const auto& im_info_tensor = *ctx->Input<Tensor>(2);
  const auto& anchors_tensor = *ctx->Input<Tensor>(3);

  ORT_ENFORCE_EQ(scores.Shape().NumDimensions(), 4);
  //ORT_ENFORCE(scores.template IsType<float>(), scores.dtype().name()); // TODO
  const auto num_images = scores.Shape().GetDims()[0];
  const auto A = scores.Shape().GetDims()[1];
  const auto height = scores.Shape().GetDims()[2];
  const auto width = scores.Shape().GetDims()[3];
  const auto box_dim = anchors_tensor.Shape().GetDims()[1];
  ORT_ENFORCE(box_dim == 4 || box_dim == 5);

  // bbox_deltas: (num_images, A * box_dim, H, W)
  ORT_ENFORCE_EQ(
      bbox_deltas.Shape().GetDims(),
      (std::vector<int64_t>{num_images, box_dim * A, height, width}));

  // im_info_tensor: (num_images, 3), format [height, width, scale; ...]
  ORT_ENFORCE_EQ(im_info_tensor.Shape().GetDims(), (std::vector<int64_t>{num_images, 3}));
  // ORT_ENFORCE(
  //     im_info_tensor.template IsType<float>(), im_info_tensor.dtype().name()); // TODO

  // anchors: (A, box_dim)
  ORT_ENFORCE_EQ(anchors_tensor.Shape().GetDims(), (std::vector<int64_t>{A, box_dim}));
  // ORT_ENFORCE(
  //     anchors_tensor.template IsType<float>(), anchors_tensor.dtype().name()); // TODO

  Eigen::Map<const ERArrXXf> im_info(
      im_info_tensor.Data<float>(),
      im_info_tensor.Shape().GetDims()[0],
      im_info_tensor.Shape().GetDims()[1]);

  Eigen::Map<const ERArrXXf> anchors(
      anchors_tensor.Data<float>(),
      anchors_tensor.Shape().GetDims()[0],
      anchors_tensor.Shape().GetDims()[1]);

  std::vector<ERArrXXf> im_boxes(num_images);
  std::vector<EArrXf> im_probs(num_images);
  for (int i = 0; i < num_images; i++) {
    auto cur_im_info = im_info.row(i);
    auto cur_bbox_deltas = GetSubTensorView<float>(bbox_deltas, i);
    auto cur_scores = GetSubTensorView<float>(scores, i);

    ERArrXXf& im_i_boxes = im_boxes[i];
    EArrXf& im_i_probs = im_probs[i];
    ProposalsForOneImage(
        cur_im_info,
        anchors,
        cur_bbox_deltas,
        cur_scores,
        &im_i_boxes,
        &im_i_probs);
  }

  int64_t roi_counts = 0;
  for (int i = 0; i < num_images; i++) {
    roi_counts += im_boxes[i].rows();
  }
  const int64_t roi_col_count = box_dim + 1;
  auto* out_rois = ctx->Output(0, {roi_counts, roi_col_count});
  auto* out_rois_probs = ctx->Output(1, {roi_counts});
  float* out_rois_ptr = out_rois->template MutableData<float>();
  float* out_rois_probs_ptr = out_rois_probs->template MutableData<float>();
  for (int i = 0; i < num_images; i++) {
    const ERArrXXf& im_i_boxes = im_boxes[i];
    const EArrXf& im_i_probs = im_probs[i];
    int64_t csz = im_i_boxes.rows();

    // write rois
    Eigen::Map<ERArrXXf> cur_rois(out_rois_ptr, csz, roi_col_count);
    cur_rois.block(0, 1, csz, box_dim) = im_i_boxes;

    // write rois_probs
    Eigen::Map<EArrXf>(out_rois_probs_ptr, csz) = im_i_probs;

    out_rois_ptr += csz * roi_col_count;
    out_rois_probs_ptr += csz;
  }

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime

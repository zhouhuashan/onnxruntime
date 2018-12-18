// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/dynamic_slice.h"
#include <stack>

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(
    DynamicSlice,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T",    DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(),DataTypeImpl::GetTensorType<int64_t>()}),
    DynamicSlice);

template<typename Tind>
Status DynamicSliceBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {

  auto data_tensor   = context->Input<Tensor>(0);
  auto starts_tensor = context->Input<Tensor>(1);
  auto ends_tensor   = context->Input<Tensor>(2);
  auto axes_tensor   = context->Input<Tensor>(3);
  ORT_ENFORCE(data_tensor   != nullptr);
  ORT_ENFORCE(starts_tensor != nullptr);
  ORT_ENFORCE(ends_tensor   != nullptr);

  auto data_shape    = data_tensor->Shape();
  auto starts_shape  = starts_tensor->Shape();
  auto ends_shape    = ends_tensor->Shape();
  if (data_shape.NumDimensions() == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "data tensor needs to be an array.");
  } else if (starts_shape.NumDimensions() * ends_shape.NumDimensions() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
      "starts tensor and ends tensor both need to be 1-D array.");
  } else if (starts_shape[0] > static_cast<int64_t>(data_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
      "starts tensor has more indices than dimension of data tensor");
  } else if (ends_shape[0] > static_cast<int64_t>(data_shape.NumDimensions())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
      "ends tensor has more indices than dimension of data tensor");
  } else if (starts_shape != ends_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
      "starts tensor and ends tensor need to have same shape.");
  }

  auto start_indices = static_cast<const Tind*>(starts_tensor->DataRaw());
  auto end_indices   = static_cast<const Tind*>(ends_tensor->DataRaw());

  std::vector< typename std::pair<int64_t, int64_t> > boundaries;
#pragma omp parallel for
  for (uint64_t i = 0; i < data_shape.NumDimensions(); ++i) {
    boundaries.push_back(std::make_pair(0, data_shape[i]));
  }
  if (axes_tensor == nullptr) {  
    for (int64_t i = 0; i < starts_shape.Size(); ++i) {
      auto lowerBound = start_indices[i] < 0 ? data_shape[i] + start_indices[i] : start_indices[i];
      auto upperBound = end_indices[i]   < 0 ? data_shape[i] + end_indices[i]   : end_indices[i];
      if (lowerBound > boundaries[i].first) {
        boundaries[i].first = lowerBound;
      }
      if (upperBound < boundaries[i].second) {
        boundaries[i].second = upperBound;
      }
      if (boundaries[i].first >= boundaries[i].second) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
          "found wrong start and end indice, start = ", start_indices[i], "end = ", end_indices[i]);
      } 
    }
  } else {
    auto axes_shape = axes_tensor->Shape();
    if (axes_shape.NumDimensions() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
        "axes tensor and ends tensor both need to be 1-D array.");
    } else if (axes_shape != starts_shape) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
        "axes tensor and indices tensor must have same shape.");
    } else if (axes_shape.Size() > static_cast<int64_t>(data_shape.NumDimensions())) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
        "axes tensor has more indice than dimension of data tensor.");
    }
    auto axes = static_cast<const Tind*>(axes_tensor->DataRaw());
    for (int64_t i; i < axes_shape.Size(); ++i) {
      auto axis = axes[i];
      if (axis >= static_cast<int64_t>(boundaries.size())) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
          "found wrong axis ", axis);
      }
      auto lowerBound = start_indices[i] < 0 ? data_shape[axis] + start_indices[i] : start_indices[i];
      auto upperBound = end_indices[i]   < 0 ? data_shape[axis] + end_indices[i]   : end_indices[i];
      if (lowerBound > boundaries[axis].first) {
        boundaries[axis].first = lowerBound;
      }
      if (upperBound < boundaries[axis].second) {
        boundaries[axis].second = upperBound;
      }
      if (boundaries[axis].first >= boundaries[axis].second) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
          "found wrong start and end indice, start = ", start_indices[i], "end = ", end_indices[i]);
      } 
    }
  }
  std::vector<int64_t> output_shape;
  output_shape.reserve(boundaries.size());
  for (auto& boundary: boundaries) {
    output_shape.push_back(boundary.second - boundary.first);
  }
  auto output_tensor = context->Output(0, TensorShape(output_shape));
  if (data_tensor->DataType() == DataTypeImpl::GetType<std::string>()) {
    p.input_str_base  = static_cast<const std::string*>(data_tensor->DataRaw());
    p.output_str_base = static_cast<std::string*>(output_tensor->MutableDataRaw());
  } else {
    p.input_base      = static_cast<const uint8_t*>(data_tensor->DataRaw());
    p.output_base     = static_cast<uint8_t*>(output_tensor->MutableDataRaw());
  }

  p.element_bytes     = data_tensor->DataType()->Size();
  p.element_to_copy   = boundaries.back().second - boundaries.back().first;
  p.bytes_to_copy     = p.element_to_copy * p.element_bytes;

  std::stack< typename std::pair<int64_t,int64_t> > stk;
  for (int64_t i = boundaries[0].first; i < boundaries[0].second; ++i) {
    stk.push(std::make_pair(i * data_shape.SizeFromDimension(1), 1));
  }
  while (!stk.empty()) {
    std::pair<int64_t, int64_t> top = std::move(stk.top());
    stk.pop();
    if (top.second == static_cast<int64_t>(boundaries.size())) {
      p.element_offsets.push_back(top.first);
    } else {
      for (int64_t i = boundaries[top.second].first; i < boundaries[top.second].second; ++i) {
        stk.push(std::make_pair(top.first + i * data_shape.SizeFromDimension(top.second + 1), top.second + 1));
      }
    }
  }
  return Status::OK();
}

template Status DynamicSliceBase::PrepareForCompute<int32_t>(OpKernelContext*, Prepare&) const;
template Status DynamicSliceBase::PrepareForCompute<int64_t>(OpKernelContext*, Prepare&) const;

Status DynamicSlice::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(context->Input<Tensor>(1)->DataType() == DataTypeImpl::GetType<int32_t>() ? 
                      PrepareForCompute<int32_t>(context, p) : PrepareForCompute<int64_t>(context, p));
  return nullptr == p.input_str_base ? SliceNumber(p) : SliceString(p);
}

Status DynamicSlice::SliceNumber(const Prepare& p) const {
#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(p.element_offsets.size()); ++i) {
    memcpy(p.output_base + i * p.bytes_to_copy,
           p.input_base + p.element_offsets[i] * p.element_bytes,
           p.bytes_to_copy);
  }
  return Status::OK();
}

Status DynamicSlice::SliceString(const Prepare& p) const {
#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(p.element_offsets.size()); ++i) {
    for (int64_t j = 0; j < static_cast<int64_t>(p.element_to_copy); ++j) {
      p.output_str_base[i * p.element_to_copy + j] = p.input_str_base[p.element_offsets[i] + j];
    }
  }
  return Status::OK();
}

}



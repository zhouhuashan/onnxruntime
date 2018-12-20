// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/dynamic_slice.h"
#include <stack>
#include <iostream>
#define INPUTS_TENSOR_SHAPE_INVALID "data tensor needs to be an array" 
#define STARTS_TENSOR_SHAPE_INVALID "starts needs to be a 1-D array."
#define INDICE_TENSOR_SHAPE_NOTSAME "starts tensor and ends tensor need to have same shape."
using DIMS = std::vector<int64_t>;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    DynamicSlice,
    9,
    KernelDefBuilder()
        .TypeConstraint("T",    DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(),DataTypeImpl::GetTensorType<int64_t>()}),
    DynamicSlice);

void AssignDimension(int64_t& output, int64_t start, int64_t end) {
  start = start < 0 ? output + start : start;
  end   = end   < 0 ? output + end   : end;
  auto diff = end - start;
  if (diff < 0 || diff > output) output = 0;
  else output = diff;
}

template<typename Tind>
int64_t AdjustOutputShape (DIMS& outputs, const DIMS& starts, const DIMS& ends, const Tensor* tensor, std::string& err) {
  int64_t slice_stop_at = starts.size();
  std::stringstream err_stream;
  if (nullptr == tensor) {
#pragma omp parallel for
    for (size_t i = 0; i < starts.size(); ++i) {
      AssignDimension(outputs[i], starts[i], ends[i]);
      if (0 == outputs[i]) {
        err_stream << "Found err start and end: [" << starts[i] << "," << ends[i] << "].";
      }
    }
  } else {
    DIMS axes(tensor->Data<Tind>(), tensor->Data<Tind>() + tensor->Shape().Size());
    if (axes.size() <= outputs.size() && axes.size() == starts.size()) {
      slice_stop_at = axes.back() + 1;
#pragma omp parallel for
      for (size_t i = 0; i < axes.size(); ++i) {
        auto axis = axes[i];
        if (axis < static_cast<int64_t>(outputs.size())) {
          AssignDimension(outputs[axis], starts[i], ends[i]);
          if (outputs[axis] == 0) {
            err_stream << "Found err start and end: [" << starts[i] << "," << ends[i] << "].";
          }
        } 
      }
    } else err_stream << "Number of axes is invalid";
  }
  err = std::move(err_stream.str());
  return slice_stop_at;
}

template<typename Tind>
void FindAllOffset(std::vector<uint64_t>& offsets,
		   const DIMS& data_shape,
		   const DIMS& output_shape,
		   const DIMS& starts,
		   const Tensor* tensor,
		   int64_t slice_stop_at)
{
  DIMS merged_starts;
  if (nullptr == tensor) {
    merged_starts = starts;
  } else {
    merged_starts.assign(output_shape.size(),0);
    DIMS axes(tensor->Data<Tind>(), tensor->Data<Tind>() + tensor->Shape().Size());
#pragma omp parallel for
    for (size_t i = 0; i < axes.size(); ++i) {
      merged_starts[axes[i]] = starts[i];
    }
  }
  std::vector<int64_t> sizeFromDim(data_shape);
  sizeFromDim.push_back(1);
  for (int64_t i = static_cast<int64_t>(sizeFromDim.size()) - 2; i >= 0; --i) {
    sizeFromDim[i] *= sizeFromDim[i+1];
  }
  std::stack< typename std::pair<int32_t, int32_t> > stk;
  for (int64_t i = merged_starts[0] + output_shape[0] - 1; i >= merged_starts[0]; --i) {
    stk.push(std::make_pair(i*sizeFromDim[1], 1));
  }
  while (!stk.empty()) {
     std::pair<int32_t, int32_t> top = std::move(stk.top());
     stk.pop();
     if (top.second == slice_stop_at) {
       offsets.push_back(top.first);
     } else {
       for (int64_t i = merged_starts[top.second] + output_shape[top.second] - 1; i >= merged_starts[top.second]; --i) {
         stk.push(std::make_pair(top.first + i * sizeFromDim[top.second + 1], top.second + 1));
       }
     }//else
  }//while
}

template<typename Tind>
Status DynamicSliceBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {

  ORT_ENFORCE(nullptr != context->Input<Tensor>(0));
  ORT_ENFORCE(nullptr != context->Input<Tensor>(1));
  ORT_ENFORCE(nullptr != context->Input<Tensor>(2));
  auto data_tensor     = context->Input<Tensor>(0);
  auto starts_tensor   = context->Input<Tensor>(1);
  auto ends_tensor     = context->Input<Tensor>(2);
  auto data_shape      = data_tensor->Shape();
  auto starts_shape    = starts_tensor->Shape();
  auto ends_shape      = ends_tensor->Shape();
  auto data_rank       = data_shape.NumDimensions();
  auto starts_rank     = starts_shape.NumDimensions();
  if (data_rank == 0)             return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, INPUTS_TENSOR_SHAPE_INVALID);
  if (starts_rank != 1)           return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, STARTS_TENSOR_SHAPE_INVALID);
  if (starts_shape != ends_shape) return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, INDICE_TENSOR_SHAPE_NOTSAME);
  DIMS starts (starts_tensor->Data<Tind>(), starts_tensor->Data<Tind>() + starts_shape.Size());
  DIMS ends   (ends_tensor->Data<Tind>(),   ends_tensor->Data<Tind>()   + ends_shape.Size());

  std::string err;
  DIMS output_shape(data_shape.GetDims());
  auto slice_stop_at   = AdjustOutputShape<Tind>(output_shape, starts, ends,
                                                 context->Input<Tensor>(3), err);
  if (err.size() > 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, err.c_str());
  }

  p.element_bytes      = data_tensor->DataType()->Size();
  p.element_to_copy    = data_shape.SizeFromDimension(slice_stop_at);
  p.bytes_to_copy      = p.element_to_copy * p.element_bytes;
  auto output_tensor   = context->Output(0, TensorShape(output_shape));
  if (data_tensor->DataType() == DataTypeImpl::GetType<std::string>()) {
    p.input_str_base   = static_cast<const std::string*>(data_tensor->DataRaw());
    p.output_str_base  = static_cast<std::string*>(output_tensor->MutableDataRaw());
  } else {
    p.input_base       = static_cast<const uint8_t*>(data_tensor->DataRaw());
    p.output_base      = static_cast<uint8_t*>(output_tensor->MutableDataRaw());
  }

  FindAllOffset<Tind>(p.element_offsets,
		      data_shape.GetDims(),
		      output_shape, starts,
	       	      context->Input<Tensor>(3),
		      slice_stop_at);
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

}//namespace onnxruntime



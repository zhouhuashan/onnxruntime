//https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather
#include "core/providers/cpu/tensor/gather.h"
#include "core/inc/op_kernel_author.h"
#include "core/common/common.h"

namespace Lotus {

ONNX_CPU_OPERATOR_KERNEL(
    Gather,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes())
                      .TypeConstraint("Tind", std::vector<MLDataType>{
                                               DataTypeImpl::GetTensorType<int32_t>(),
                                               DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

Status GatherBase::PrepareForCompute(OpKernelContext* context, Prepare& p) const {
  p.input_tensor = context->Input<Tensor>(0);
  const TensorShape& input_data_shape = p.input_tensor->Shape();
  p.indices_tensor = context->Input<Tensor>(1);
  const TensorShape& indices_shape = p.indices_tensor->Shape();

  p.axis = HandleNegativeAxis(axis_, input_data_shape.NumDimensions());

  std::vector<int64_t> shape(indices_shape.GetDims().begin(), indices_shape.GetDims().end());
  shape.insert(shape.begin(), input_data_shape.GetDims().begin(), input_data_shape.GetDims().begin() + p.axis);
  shape.insert(shape.end(), input_data_shape.GetDims().begin() + p.axis + 1, input_data_shape.GetDims().end());

  p.output_tensor = context->Output(0, TensorShape(shape));

  return Status::OK();
}

Status Gather::Compute(OpKernelContext* context) const {
  Prepare p;
  LOTUS_RETURN_IF_ERROR(PrepareForCompute(context, p));

  const TensorShape& input_data_shape = p.input_tensor->Shape();

  auto is_string_type = p.input_tensor->DataType() == DataTypeImpl::GetType<std::string>();
  
  size_t element_bytes = p.input_tensor->DataType()->Size();
  const int64_t block = input_data_shape.SizeFromDimension(p.axis + 1);
  const int64_t block_size = block * element_bytes;
  const int64_t M = input_data_shape.SizeToDimension(p.axis);
  const int64_t N = p.indices_tensor->Shape().Size();
  const int64_t data_batch_bytes = input_data_shape.SizeFromDimension(p.axis) * element_bytes;
  const int64_t gathered_batch_bytes = N * block * element_bytes;

  size_t idx_bytes = p.indices_tensor->DataType()->Size();
  const uint8_t* idxs_base = static_cast<const uint8_t*>(p.indices_tensor->DataRaw());
  const uint8_t* src_base = static_cast<const uint8_t*>(p.input_tensor->DataRaw());
  uint8_t* dst_base = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw());

  for (int64_t batch = 0; batch < M; ++batch) {
    const int64_t src_offset_batch = batch * data_batch_bytes;
    const int64_t dst_offset_batch = batch * gathered_batch_bytes;
    for (int64_t i = 0; i < N; ++i) {
      int64_t idx = static_cast<int64_t>(*(idxs_base + i * idx_bytes));
      if (idx < 0 || idx >= input_data_shape[p.axis]) {
        return LOTUS_MAKE_STATUS(LOTUS, INVALID_ARGUMENT, "indices element out of data bounds, idx=", idx,
                                 " data_dim=", input_data_shape[p.axis]);
      }
      const int64_t src_offset = src_offset_batch + idx * block_size;
      const int64_t dst_offset = dst_offset_batch + i * block_size;

      if (is_string_type) {
        reinterpret_cast<std::string*>(dst_base)[dst_offset/element_bytes] =
            reinterpret_cast<const std::string*>(src_base)[src_offset/element_bytes];
      } else {
        memcpy(dst_base + dst_offset, src_base + src_offset, block_size);
      }
    }
  }

  return Status::OK();
}

}  // namespace Lotus

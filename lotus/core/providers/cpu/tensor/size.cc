#include "core/providers/cpu/tensor/size.h"

namespace Lotus {

Status Size::Compute(OpKernelContext* ctx) const {
  const Tensor& input_tensor = *ctx->Input<Tensor>(0);

  std::vector<int64_t> empty_list_of_dimensions;
  TensorShape scalar_shape(empty_list_of_dimensions);
  Tensor* p_output_tensor = ctx->Output(0, scalar_shape);
  int64_t* p_output_scalar = p_output_tensor->MutableData<int64_t>();

  *p_output_scalar = input_tensor.Shape().Size();

  return Status::OK();
}

// The implementation of Size works for tensors of any type. The types listed below are
// based on the ones the datatypes in data_types.cc.
// TODO: we should not have to add the TypeConstraint below, since it is meant to be in
// addition to the ONNX specification. But the registration doesn't seem to work if we
// omit this.
// TODO: Both Lotus and ONNX lists of types seem somewhat incomplete and incomparable.

REGISTER_KERNEL(KernelDefBuilder("Size")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T",
                                    std::vector<MLDataType>({DataTypeImpl::GetTensorType<float>(),
                                                             DataTypeImpl::GetTensorType<double>(),
                                                             DataTypeImpl::GetTensorType<int16_t>(),
                                                             DataTypeImpl::GetTensorType<int>(),
                                                             DataTypeImpl::GetTensorType<int64_t>(),
                                                             DataTypeImpl::GetTensorType<uint8_t>(),
                                                             DataTypeImpl::GetTensorType<uint16_t>(),
                                                             DataTypeImpl::GetTensorType<uint32_t>(),
                                                             DataTypeImpl::GetTensorType<uint64_t>(),
                                                             DataTypeImpl::GetTensorType<bool>()})),
                Size);

}  // namespace Lotus

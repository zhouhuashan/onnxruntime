#include "core/providers/cpu/tensor/transpose.h"

namespace Lotus {

/* A permutation [a,b,c,...] indicates that 
   - The 0-th dimension of the output corresponds to the a-th dimension of input
   - The 1-st dimension of the output corresponds to the b-th dimension of input
   - The 2-nd dimension of the output corresponds to the c-th dimension of input
   etc.
   */

// The following is a reference (unoptimized) implementation of Transpose.
// TODO: Optimize the implementation to use memcpy for sub-blocks that can be so copied.

template <>
Status Transpose<float>::compute(OpKernelContext* ctx) const {
  const Tensor& X = *ctx->input<Tensor>(0);
  const TensorShape& input_shape = X.shape();
  const std::vector<int64_t>& input_dims = input_shape.GetDims();
  size_t rank = input_dims.size();

  // Determine permutation to use:
  // If no permutation was specified in the attributes, the default is [rank-1, ..., 0]
  const std::vector<int64_t>* p_perm;
  std::vector<int64_t> default_perm(rank);

  if (perm_specified_)
    p_perm = &perm_;
  else {
    for (int i = 0; i < rank; ++i)
      default_perm[i] = rank - i - 1;
    p_perm = &default_perm;
  }

  // Determine shape of output, as well as stride to be used:
  // stride[i] indicates the stride for the input-tensor dimension corresponding
  // to the i-th dimension of the output

  std::vector<int64_t> output_dims(rank);
  std::vector<size_t> stride(rank);
  for (int i = 0; i < rank; i++) {
    size_t inpdim = (*p_perm)[i];
    output_dims[i] = input_dims[inpdim];
    if (inpdim + 1 < rank)
      stride[i] = input_shape.SizeFromDimension(inpdim + 1);
    else
      stride[i] = 1;
  }

  TensorShape output_shape{output_dims};
  Tensor* Y = ctx->output(0, output_shape);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  auto size = output_shape.Size();
  std::vector<int64_t> y_index(rank, 0);  // index used to iterate over Y's iteration-space
  for (size_t i = 0; i < size; ++i) {
    // convert y_index into offset in X's data
    size_t x_offset = 0;
    for (int j = 0; j < rank; ++j) {
      x_offset += y_index[j] * stride[j];
    }
    // copy
    LOTUS_ENFORCE((0 <= x_offset) && (x_offset < size));
    *(Ydata + i) = *(Xdata + x_offset);
    // increment y_index:
    for (int64_t k = rank - 1; k >= 0; --k) {
      y_index[k]++;
      if (y_index[k] < output_dims[k]) break;
      y_index[k] = 0;
    }
  }

  return Status::OK();
}

REGISTER_KERNEL(KernelDefBuilder("Transpose")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1, 2)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                Transpose<float>);

}  // namespace Lotus

#include "core/providers/cpu/reduction/reduction_ops.h"
#include "core/util/math_cpuonly.h"
using namespace std;
namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("ReduceL1")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ReduceL1<float>);

REGISTER_KERNEL(KernelDefBuilder("ReduceL2")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ReduceL2<float>);

REGISTER_KERNEL(KernelDefBuilder("ReduceLogSum")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ReduceLogSum<float>);

REGISTER_KERNEL(KernelDefBuilder("ReduceLogSumExp")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ReduceLogSumExp<float>);

REGISTER_KERNEL(KernelDefBuilder("ReduceMax")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ReduceMax<float>);

REGISTER_KERNEL(KernelDefBuilder("ReduceMean")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ReduceMean<float>);

REGISTER_KERNEL(KernelDefBuilder("ReduceMin")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ReduceMin<float>);

REGISTER_KERNEL(KernelDefBuilder("ReduceProd")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ReduceProd<float>);

REGISTER_KERNEL(KernelDefBuilder("ReduceSum")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ReduceSum<float>);

REGISTER_KERNEL(KernelDefBuilder("ReduceSumSquare")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ReduceSumSquare<float>);

REGISTER_KERNEL(KernelDefBuilder("ArgMax")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ArgMax<float>);

REGISTER_KERNEL(KernelDefBuilder("ArgMin")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                ArgMin<float>);
template <bool allow_multi_axes>
void ReduceKernel<allow_multi_axes>::PrepareForReduce(OpKernelContext* ctx,
                                                      std::vector<float>& transposedInputData,
                                                      Tensor** reducedTensor,
                                                      int64_t& block_size,
                                                      int64_t& blocks) const {
  const Tensor& input = *ctx->Input<Tensor>(0);

  size_t ndim = input.Shape().GetDims().size();
  for (int64_t axe : axes_) {
    LOTUS_ENFORCE(axe >= 0 && axe < (int64_t)ndim, "Axis attribute out of range");
  }

  transposedInputData.resize(input.Shape().Size(), 0);

  std::vector<int64_t> axes = axes_;
  if (axes.empty()) {
    // This is the default case for non-arg kind reductions. Reduce on all dimensions.
    for (int i = 0; i < ndim; i++)
      axes.push_back(i);
  }

  std::sort(axes.begin(), axes.end());

  vector<bool> keep_axis(ndim, true);
  for (auto i : axes) {
    keep_axis[i] = false;
  }

  //transpose the input so that all to-be-reduced axes are at the head
  vector<int64_t> transposed_axes(axes.begin(), axes.end());
  for (int i = 0; i < ndim; ++i) {
    if (keep_axis[i]) {
      transposed_axes.push_back(i);
    }
  }

  vector<int64_t> new_dims_(transposed_axes.size());
  for (int i = 0; i < transposed_axes.size(); ++i) {
    new_dims_[i] = input.Shape().GetDims().at(transposed_axes[i]);
  }

  int num_axes = static_cast<int>(transposed_axes.size());
  auto in_dims = input.Shape().GetDims();

  // Measure amount of contiguous data we can copy at once
  int64_t blocksize = 1;
  int n_shared_idxs = 0;
  for (int i = num_axes - 1; i >= 0; --i) {
    if (transposed_axes[i] == i) {
      blocksize *= new_dims_[i];
      ++n_shared_idxs;
    } else {
      break;
    }
  }

  const float* from_data = input.Data<float>();
  float* to_data = &transposedInputData[0];
  size_t count = input.Shape().Size();

  //set to-be-reduced axes to one. squeeze is keepdims_ is false
  int64_t first_dim = 1;
  std::vector<int64_t> reduced_dims;
  for (int i = 0; i < in_dims.size(); i++) {
    if (keep_axis[i]) {
      reduced_dims.push_back(in_dims[i]);
    } else {
      first_dim *= in_dims[i];
      if (keepdims_) {
        reduced_dims.push_back(1);
      }
    }
  }

  *reducedTensor = ctx->Output(0, reduced_dims);
  block_size = input.Shape().Size() / first_dim;
  blocks = first_dim;

  if (num_axes < 2 || n_shared_idxs == num_axes) {
    memcpy(to_data, from_data, count * sizeof(float));
    return;
  }

  int itr_axes = num_axes - n_shared_idxs;

  // Calculate strides
  std::vector<int64_t> stride_x(itr_axes, 0);
  for (size_t i = 0; i < itr_axes; i++) {
    stride_x[i] = 1;
    for (size_t j = transposed_axes[i] + 1; j < itr_axes; j++) {
      stride_x[i] *= in_dims[j];
    }
  }

  std::vector<int64_t> itr_idxs(itr_axes, 0);

  // Branch here to avoid branching within the loop
  if (blocksize > 1) {
    for (size_t index = 0; index < (count / blocksize); index++) {
      int64_t from_index = 0;
      for (int i = 0; i < itr_axes; ++i) {
        from_index += stride_x[i] * itr_idxs[i];
      }

      memcpy(
          to_data + blocksize * index,
          from_data + blocksize * from_index,
          blocksize * sizeof(float));

      ++itr_idxs[itr_axes - 1];
      for (int i = itr_axes - 1; i >= 1; --i) {
        auto expected_dim = new_dims_[i];
        if (itr_idxs[i] < expected_dim) {
          break;
        }
        itr_idxs[i] %= expected_dim;
        ++itr_idxs[i - 1];
      }
    }
  } else {
    for (size_t index = 0; index < count; index++) {
      int64_t from_index = 0;
      for (int i = 0; i < itr_axes; ++i) {
        from_index += stride_x[i] * itr_idxs[i];
      }

      *(to_data + index) = *(from_data + from_index);

      ++itr_idxs[itr_axes - 1];
      for (int i = itr_axes - 1; i >= 1; --i) {
        auto expected_dim = new_dims_[i];
        if (itr_idxs[i] < expected_dim) {
          break;
        }
        itr_idxs[i] %= expected_dim;
        ++itr_idxs[i - 1];
      }
    }
  }
}

template <>
Status ReduceL1<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  float* output_data = reduced->MutableData<float>();

  EigenVectorMap<float> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks).cwiseAbs().rowwise().sum();

  return Status::OK();
}

template <>
Status ReduceL2<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  float* output_data = reduced->MutableData<float>();

  EigenVectorMap<float> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks).rowwise().norm();

  return Status::OK();
}

template <>
Status ReduceLogSum<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  float* output_data = reduced->MutableData<float>();

  EigenVectorMap<float> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks).rowwise().sum();
  for (int j = 0; j < block_size; ++j) {
    *(output_data) = std::log(*(output_data));
    ++output_data;
  }

  return Status::OK();
}

template <>
Status ReduceLogSumExp<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  float* output_data = reduced->MutableData<float>();

  for (int j = 0; j < block_size; ++j) {
    float max_value = std::numeric_limits<float>::lowest();
    for (int i = 0; i < blocks; ++i) {
      max_value = std::max(max_value, transposedInputData[i * block_size + j]);
    }
    float scaled_exp_sum = 0;
    for (int i = 0; i < blocks; ++i) {
      scaled_exp_sum += std::exp(transposedInputData[i * block_size + j] - max_value);
    }
    *(output_data++) = std::log(scaled_exp_sum) + max_value;
  }
  return Status::OK();
}

template <>
Status ReduceMax<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  float* output_data = reduced->MutableData<float>();

  EigenVectorMap<float> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks).rowwise().maxCoeff();

  return Status::OK();
}

template <>
Status ReduceMean<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  float* output_data = reduced->MutableData<float>();

  EigenVectorMap<float> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks).rowwise().mean();

  return Status::OK();
}

template <>
Status ReduceMin<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  float* output_data = reduced->MutableData<float>();

  EigenVectorMap<float> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks).rowwise().minCoeff();

  return Status::OK();
}

template <>
Status ReduceProd<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  float* output_data = reduced->MutableData<float>();

  EigenVectorMap<float> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks).rowwise().prod();

  return Status::OK();
}

template <>
Status ReduceSum<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  float* output_data = reduced->MutableData<float>();

  EigenVectorMap<float> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks).rowwise().sum();

  return Status::OK();
}

template <>
Status ReduceSumSquare<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  float* output_data = reduced->MutableData<float>();

  EigenVectorMap<float> out_vec(output_data, block_size);
  out_vec = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks).rowwise().squaredNorm();

  return Status::OK();
}

template <>
Status ArgMax<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  int64_t* output_data = reduced->MutableData<int64_t>();

  Eigen::MatrixXf::Index maxIndex;
  auto matrixData = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks);
  for (int i = 0; i < block_size; ++i) {
    matrixData.row(i).maxCoeff(&maxIndex);
    *(output_data++) = maxIndex;
  }

  return Status::OK();
}

template <>
Status ArgMin<float>::Compute(OpKernelContext* ctx) const {
  std::vector<float> transposedInputData;
  int64_t block_size, blocks;
  Tensor* reduced;
  PrepareForReduce(ctx, transposedInputData, &reduced, block_size, blocks);

  int64_t* output_data = reduced->MutableData<int64_t>();

  Eigen::MatrixXf::Index minIndex;
  auto matrixData = ConstEigenMatrixMap<float>(&transposedInputData[0], block_size, blocks);
  for (int i = 0; i < block_size; ++i) {
    matrixData.row(i).minCoeff(&minIndex);
    *(output_data++) = minIndex;
  }

  return Status::OK();
}

}  // namespace Lotus

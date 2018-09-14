// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "upsample.h"

#ifdef USE_TVM
#include <tvm/tvm.h>
#include <tvm/build_module.h>
#include <tvm/runtime/ndarray.h>
#include <topi/elemwise.h>
#include <topi/nn/upsampling.h>
#include <topi/cuda/injective.h>
#include <algorithm>

using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Upsample,                                                   \
      kOnnxDomain,                                                \
      7,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Upsample<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(int32_t)

struct TVMState {
  //key
  std::vector<int64_t> last_x_dims;

  //values
  std::vector<int64_t> x_tvm_dims;
  std::vector<int64_t> y_tvm_dims;
  tvm::runtime::PackedFunc func;
};

template <typename T>
Status Upsample<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const std::vector<int64_t>& X_dims = X->Shape().GetDims();
  auto rank = X_dims.size();
  if (rank == 0)
    return Status(LOTUS, INVALID_ARGUMENT, "Upsample: input tensor cannot be scalar.");

  if (rank != scales_.size())
    return Status(LOTUS, INVALID_ARGUMENT, "Upsample: input tensor's dimension does not match the scales.");

  std::vector<int64_t> Y_dims;
  for (std::size_t i = 0; i < X_dims.size(); i++) {
    Y_dims.push_back(static_cast<int64_t>(scales_[i] * X_dims[i]));
  }
  Tensor* Y = context->Output(0, Y_dims);
  typedef typename ToCudaType<T>::MappedType CudaT;

  // TVM is not thread-safe
  std::lock_guard<std::mutex> lock(mutex);

  if (s_ == nullptr)
    s_ = std::make_unique<TVMState>();

  if (s_->last_x_dims != X_dims) {
    s_->last_x_dims = X_dims;
    // TOPI upsampling only works in HW dimension for NCHW input, so fuse/pad dims to 4D
    auto MakeNCHW = [](const std::vector<int64_t> dims) -> std::vector<int64_t> {
      std::vector<int64_t> tvm_dims(4, 1);
      int rank = gsl::narrow_cast<int>(dims.size());
      int actual_out_rank = std::min(rank, 4);
      int padded = 4 - actual_out_rank;
      int fused = std::max(rank - 4, 0);
      for (int i = 0; i < actual_out_rank; i++)
        tvm_dims[padded + i] = dims[fused + i];

      if (rank > 4)
        tvm_dims[0] = TensorShape(dims).SizeToDimension(rank - 3);

      return tvm_dims;
    };

    s_->x_tvm_dims = MakeNCHW(X_dims);
    s_->y_tvm_dims = MakeNCHW(Y_dims);

    tvm::Array<tvm::Expr> a_shape;
    for (auto xd : s_->x_tvm_dims)
      a_shape.push_back(tvm::Expr(gsl::narrow_cast<int32_t>(xd)));

    tvm::Array<tvm::Expr> out_hw;
    out_hw.push_back(tvm::Expr(gsl::narrow_cast<int32_t>(s_->y_tvm_dims[2])));
    out_hw.push_back(tvm::Expr(gsl::narrow_cast<int32_t>(s_->y_tvm_dims[3])));

    tvm::Tensor tvm_X;
    tvm::Tensor tvm_Y;

    if (std::is_same<T, float>::value) {
      tvm_X = tvm::placeholder(a_shape, tvm::Float(32), "A");
    } else if (std::is_same<T, int32_t>::value) {
      tvm_X = tvm::placeholder(a_shape, tvm::Int(32), "A");
    } else
      return Status(LOTUS, FAIL, "Unsupported data type");

    switch (mode_) {
      case UpsampleMode::NN:
        tvm_Y = topi::nn::upsampling(tvm_X, out_hw, "NCHW", "NEAREST_NEIGHBOR");
        break;
      case UpsampleMode::LINEAR:
        tvm_Y = topi::nn::upsampling(tvm_X, out_hw, "NCHW", "BILINEAR");
        break;
      default:
        return Status(LOTUS, FAIL, "Upsample: unexpected mode");
    }

    auto target = tvm::target::cuda();
    auto S = topi::cuda::schedule_injective(target, {tvm_Y});

    auto args = tvm::Array<tvm::Tensor>({tvm_X, tvm_Y});
    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
    auto config = tvm::build_config();
    auto lowered = tvm::lower(S, args, "upsample", binds, config);
    auto module = tvm::build(lowered, target, tvm::target::stackvm(), config);
    s_->func = module.GetFunction("upsample");
  }

  auto data_x = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto data_y = reinterpret_cast<CudaT*>(Y->MutableData<T>());

  DLDataType dtype;
  dtype.lanes = 1;
  if (std::is_same<T, float>::value) {
    dtype.code = kDLFloat;
    dtype.bits = 32;
  } else if (std::is_same<T, int32_t>::value) {
    dtype.code = kDLInt;
    dtype.bits = 32;
  } else
    return Status(LOTUS, FAIL, "Unsupported data type");

  DLContext ctx;
  ctx.device_type = DLDeviceType::kDLGPU;
  ctx.device_id = 0;

  DLTensor dltensor_X = {const_cast<CudaT*>(data_x), ctx, gsl::narrow_cast<int>(s_->x_tvm_dims.size()), dtype, s_->x_tvm_dims.data(), nullptr, 0};
  DLTensor dltensor_Y = {data_y, ctx, gsl::narrow_cast<int>(s_->y_tvm_dims.size()), dtype, s_->y_tvm_dims.data(), nullptr, 0};

  TVMValue lvalues[2];
  int type_codes[] = {kNDArrayContainer, kNDArrayContainer};
  lvalues[0].v_handle = &dltensor_X;
  lvalues[1].v_handle = &dltensor_Y;

  tvm::TVMArgs tvm_args(lvalues, type_codes, 2);
  tvm::TVMRetValue rvalue;
  s_->func.CallPacked(tvm_args, &rvalue);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
#endif

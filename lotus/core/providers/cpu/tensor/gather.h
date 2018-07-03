#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace Lotus {
template <typename T, typename TInd>
class Gather final : public OpKernel {
 public:
  Gather(const OpKernelInfo& info) : OpKernel(info) {
    LOTUS_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* data = context->Input<Tensor>(0);
    const TensorShape& data_shape = data->Shape();
    const Tensor* indices = context->Input<Tensor>(1);
    const TensorShape& indices_shape = indices->Shape();

    const int64_t axis = HandleNegativeAxis(axis_, data_shape.NumDimensions());

    std::vector<int64_t>
        shape(indices_shape.GetDims().begin(), indices_shape.GetDims().end());
    shape.insert(shape.begin(), data_shape.GetDims().begin(), data_shape.GetDims().begin() + axis);
    shape.insert(shape.end(), data_shape.GetDims().begin() + axis + 1, data_shape.GetDims().end());
    Tensor* output = context->Output(0, TensorShape(shape));

    const int64_t block = data_shape.SizeFromDimension(axis + 1);
    const int64_t block_size = block * sizeof(T);
    const int64_t M = data_shape.SizeToDimension(axis);
    const int64_t N = indices_shape.Size();
    const int64_t data_batch = data_shape.SizeFromDimension(axis);
    const int64_t gathered_batch = N * data_shape.SizeFromDimension(axis + 1);

    const TInd* idxs = indices->template Data<TInd>();
    const T* src_base = data->template Data<T>();
    T* out = output->template MutableData<T>();

    for (int64_t batch = 0; batch < M; ++batch) {
      const T* src_p = src_base + batch * data_batch;
      T* dst_p = out + batch * gathered_batch;
      for (int64_t i = 0; i < N; ++i) {
        auto idx = idxs[i];
        if (idx < 0 || idx >= data_shape[axis]) {
          std::ostringstream err_str;
          err_str << "indices element out of data bounds, idx=" << to_string(idx);
          err_str << " data_dim=" << to_string(data_shape[axis]);
          return Status(LOTUS, INVALID_ARGUMENT, err_str.str());
        }
        const T* src = src_p + idx * block;
        T* dst = dst_p + i * block;
        memcpy(dst, src, block_size);
      }
    }

    return Status::OK();
  }

 private:
  int64_t axis_;
};
}  // namespace Lotus

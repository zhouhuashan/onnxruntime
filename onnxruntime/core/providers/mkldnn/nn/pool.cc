// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/mkldnn/mkldnn_common.h"
#include "core/providers/mkldnn/nn/pool.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"

namespace onnxruntime {
namespace mkl_dnn {

#define POOLING_KERNEL(op_name, data_type, pool_type, since_version)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      op_name,                                                                          \
      kOnnxDomain,                                                                      \
      since_version,                                                                    \
      data_type,                                                                        \
      kMklDnnExecutionProvider,                                                         \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Pool<data_type, pool_type>);

POOLING_KERNEL(AveragePool, float, AveragePool, 7)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1)
POOLING_KERNEL(MaxPool, float, MaxPool, 1)
POOLING_KERNEL(MaxPool, float, MaxPool, 8)
POOLING_KERNEL(GlobalMaxPool, float, MaxPool, 1)

namespace {
// Struct which encapsulates parameters for MKLDNN Pool primitive.
struct PoolParams {
  std::string op_name;
  mkldnn::memory::dims& src_dims;
  mkldnn::memory::dims& dst_dims;
  mkldnn::memory::dims& kernel;
  mkldnn::memory::dims& strides;
  mkldnn::memory::dims& padding_left;
  mkldnn::memory::dims& padding_right;
  bool count_include_pad;

  PoolParams(std::string op_name,
             mkldnn::memory::dims& src_dims, mkldnn::memory::dims& dst_dims,
             mkldnn::memory::dims& kernel, mkldnn::memory::dims& strides,
             mkldnn::memory::dims& padding_left, mkldnn::memory::dims& padding_right,
             bool count_include_pad)
      : op_name(op_name),
        src_dims(src_dims),
        dst_dims(dst_dims),
        kernel(kernel),
        strides(strides),
        padding_left(padding_left),
        padding_right(padding_right),
        count_include_pad(count_include_pad) {}

  // Used as the key for Pool Primitive Reuse Pool.
  std::string ToString() const {
    std::string key;
    key.reserve(128);
    key.append(op_name);
    AddDimsToKey(key, src_dims);
    AddDimsToKey(key, dst_dims);
    AddDimsToKey(key, kernel);
    AddDimsToKey(key, strides);
    AddDimsToKey(key, padding_left);
    AddDimsToKey(key, padding_right);
    key.append(count_include_pad ? "true" : "false");
    return key;
  }
};

template <typename T, PoolType type>
class PoolPrimitive : public PrimitiveBase {
 public:
  explicit PoolPrimitive(const PoolParams& params)
      : cpu_engine_(GetEngine()) {
    context_.stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
    if (context_.pool_fwd == nullptr) {
      Initialize(params);
    }
  }

  ~PoolPrimitive() = default;

  void Compute(const T* src_data, const T* dst_data) {
    context_.src_mem->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(const_cast<T*>(dst_data)));
    context_.stream->submit(context_.net);

    context_.src_mem->set_data_handle(nullptr);
    context_.dst_mem->set_data_handle(nullptr);
    return;
  }

  mkldnn::memory::format GetSrcMemoryFormat() const { return context_.src_fmt; }
  mkldnn::memory::format GetDstMemoryFormat() const { return context_.dst_fmt; }

  std::shared_ptr<mkldnn::memory::desc> GetDstMemoryDesc() const { return context_.dst_md; }
  std::shared_ptr<mkldnn::pooling_forward::primitive_desc>

  GetPrimitiveDesc() const {
    return context_.fwd_primitive_desc;
  }

 private:
  struct PoolContext {
    mkldnn::memory::format src_fmt;
    mkldnn::memory::format dst_fmt;

    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;

    std::shared_ptr<mkldnn::pooling_forward::desc> fwd_desc;

    std::shared_ptr<mkldnn::memory::desc> src_md;
    std::shared_ptr<mkldnn::memory::desc> dst_md;

    std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_primitive_desc;

    std::shared_ptr<mkldnn::primitive> pool_fwd;

    std::shared_ptr<mkldnn::stream> stream;
    std::vector<mkldnn::primitive> net;

    PoolContext()
        : src_fmt(mkldnn::memory::format::any),
          dst_fmt(mkldnn::memory::format::any),
          src_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          src_md(nullptr),
          fwd_primitive_desc(nullptr),
          pool_fwd(nullptr),
          stream(nullptr) {}
  };

  void Initialize(const PoolParams& params) {
    context_.src_md.reset(new mkldnn::memory::desc(
        {params.src_dims}, MklDnnType<T>(), mkldnn::memory::format::nchw));
    context_.dst_md.reset(new mkldnn::memory::desc(
        {params.dst_dims}, MklDnnType<T>(), mkldnn::memory::format::nchw));

    mkldnn::algorithm algo = mkldnn::algorithm::pooling_max;
    if (type == AveragePool) {
      algo = mkldnn::algorithm::pooling_avg_exclude_padding;
      if (params.count_include_pad) {
        algo = mkldnn::algorithm::pooling_avg_include_padding;
      }
    }
    context_.fwd_desc.reset(new mkldnn::pooling_forward::desc(
        mkldnn::prop_kind::forward_inference, algo,
        *context_.src_md, *context_.dst_md,
        params.strides, params.kernel,
        params.padding_left, params.padding_right,
        mkldnn::padding_kind::zero));

    context_.fwd_primitive_desc.reset(new mkldnn::pooling_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    context_.src_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_primitive_desc.get()->src_primitive_desc().desc().data.format);

    context_.dst_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_primitive_desc.get()->dst_primitive_desc().desc().data.format);

    context_.src_mem.reset(
        new mkldnn::memory(context_.fwd_primitive_desc.get()->src_primitive_desc(), nullptr));
    context_.dst_mem.reset(
        new mkldnn::memory(context_.fwd_primitive_desc.get()->dst_primitive_desc(), nullptr));
    context_.pool_fwd.reset(
        new mkldnn::pooling_forward(*context_.fwd_primitive_desc, *context_.src_mem, *context_.dst_mem));
    context_.net.push_back(*context_.pool_fwd);
  }

  PoolContext context_;
  mkldnn::engine& cpu_engine_;
};

// Pool which allows for reuse of MKLDNN Pool primitives which are expensive to instantiate.
// To address thread safety, the primitives are stored in a map on thread local storage.
template <typename T, PoolType type>
class PoolPrimitivePool : public PrimitivePool<T> {
 public:
  static PoolPrimitive<T, type>* Get(const PoolParams& params) {
    PoolPrimitive<T, type>* primitive = dynamic_cast<PoolPrimitive<T, type>*>(
        PoolPrimitivePool<T, type>::GetInstance().GetPrimitive(params.ToString()));
    if (primitive == nullptr) {
      auto pool_primitive = std::make_unique<PoolPrimitive<T, type>>(params);
      primitive = pool_primitive.get();
      PoolPrimitivePool<T, type>::GetInstance().SetPrimitive(params.ToString(), std::move(pool_primitive));
    }
    return primitive;
  }

 private:
  PoolPrimitivePool() = default;
  ~PoolPrimitivePool() = default;

  static PoolPrimitivePool& GetInstance() {
    static PoolPrimitivePool pool;
    return pool;
  }
};
}  // namespace

template <typename T, PoolType type>
Status Pool<T, type>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();

  if (x_shape.NumDimensions() < 3) {
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "Input dimension cannot be less than 3.");
  }

  std::vector<int64_t> kernel_shape = kernel_shape_;
  std::vector<int64_t> pads = pads_;
  std::vector<int64_t> strides = strides_;

  if (global_pooling_) {
    kernel_shape.assign(x_dims.begin() + 2, x_dims.end());
    pads.assign(kernel_shape.size(), 0);
    strides.assign(kernel_shape.size(), 1);
  }

  std::vector<int64_t> y_dims = PoolBase::SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, TensorShape(y_dims));

  const T* src_data = X->template Data<T>();
  T* dst_data = Y->template MutableData<T>();

  mkldnn::memory::dims src_dims_mkl(x_dims.begin(), x_dims.end());
  mkldnn::memory::dims dst_dims_mkl(y_dims.begin(), y_dims.end());
  mkldnn::memory::dims kernel_mkl(kernel_shape.begin(), kernel_shape.end());
  mkldnn::memory::dims strides_mkl(strides.begin(), strides.end());
  mkldnn::memory::dims padding_left_mkl(pads.begin(), pads.begin() + 2);
  mkldnn::memory::dims padding_right_mkl(pads.begin() + 2, pads.end());

  try {
    PoolParams pool_params(op_name_,
                           src_dims_mkl, dst_dims_mkl,
                           kernel_mkl, strides_mkl,
                           padding_left_mkl, padding_right_mkl,
                           count_include_pad_);
    PoolPrimitive<T, type>* pool_primitive = PoolPrimitivePool<T, type>::Get(pool_params);
    pool_primitive->Compute(src_data, dst_data);
  } catch (mkldnn::error& e) {
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "Status: ", e.status, ", message: ", e.message.c_str());
  }

  return Status::OK();
}

}  // namespace mkl_dnn
}  // namespace onnxruntime

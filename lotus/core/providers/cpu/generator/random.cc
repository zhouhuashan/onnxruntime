#include "core/providers/cpu/generator/random.h"

#include <algorithm>
#include <chrono>
#include <random>

#include "gsl/span"
namespace Lotus {

REGISTER_KERNEL(KernelDefBuilder("RandomNormal")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", std::vector<MLDataType>{
                                             DataTypeImpl::GetTensorType<float>(),
                                             DataTypeImpl::GetTensorType<double>()}),
                RandomNormal);

REGISTER_KERNEL(KernelDefBuilder("RandomUniform")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", std::vector<MLDataType>{
                                             DataTypeImpl::GetTensorType<float>(),
                                             DataTypeImpl::GetTensorType<double>()}),
                RandomUniform);

REGISTER_KERNEL(KernelDefBuilder("RandomNormalLike")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T1", DataTypeImpl::AllTensorTypes())
                    .TypeConstraint("T2", std::vector<MLDataType>{
                                              DataTypeImpl::GetTensorType<float>(),
                                              DataTypeImpl::GetTensorType<double>()}),
                RandomNormalLike);

REGISTER_KERNEL(KernelDefBuilder("RandomUniformLike")
                    .Domain(LotusIR::kOnnxDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T1", DataTypeImpl::AllTensorTypes())
                    .TypeConstraint("T2", std::vector<MLDataType>{
                                              DataTypeImpl::GetTensorType<float>(),
                                              DataTypeImpl::GetTensorType<double>()}),
                RandomUniformLike);

template <typename T, typename TDistribution>
void GenerateData(std::default_random_engine generator, TDistribution distribution, Tensor& tensor);

static Status RandomNormalCompute(float mean, float scale, std::default_random_engine generator, TensorProto::DataType dtype, Tensor& Y);
static Status RandomUniformCompute(float high, float low, std::default_random_engine generator, TensorProto::DataType dtype, Tensor& Y);

// Leaving in case we need to change to this approach
//static Status CreateOutputTensorFromTensorValues(OpKernelContext* ctx, const Tensor& X,Tensor** Y);
static Status CreateOutputTensorFromTensorShape(OpKernelContext* ctx, const Tensor& X, Tensor** Y);
static TensorProto::DataType InferDataType(const Tensor& tensor);

Status RandomNormal::Compute(OpKernelContext* ctx) const {
  Tensor& Y = *ctx->Output(0, shape_);

  auto status = RandomNormalCompute(mean_, scale_, generator_, dtype_, Y);

  return status;
}

Status RandomUniform::Compute(OpKernelContext* ctx) const {
  Tensor& Y = *ctx->Output(0, shape_);

  auto status = RandomUniformCompute(low_, high_, generator_, dtype_, Y);

  return status;
}

Status RandomNormalLike::Compute(OpKernelContext* ctx) const {
  const Tensor& X = *ctx->Input<Tensor>(0);
  Tensor* Y = nullptr;

  auto status = CreateOutputTensorFromTensorShape(ctx, X, &Y);
  LOTUS_RETURN_IF_ERROR(status);

  auto dtype = dtype_ != TensorProto_DataType_UNDEFINED ? dtype_ : InferDataType(X);

  if (dtype_ == TensorProto_DataType_UNDEFINED)
    return LOTUS_MAKE_STATUS(LOTUS, FAIL,
                             "Could not infer data type from input tensor with data type ",
                             X.DataType());

  status = RandomNormalCompute(mean_, scale_, generator_, dtype, *Y);

  return status;
}

Status RandomUniformLike::Compute(OpKernelContext* ctx) const {
  const Tensor& X = *ctx->Input<Tensor>(0);
  Tensor* Y = nullptr;

  auto status = CreateOutputTensorFromTensorShape(ctx, X, &Y);
  LOTUS_RETURN_IF_ERROR(status);

  auto dtype = dtype_ != TensorProto_DataType_UNDEFINED ? dtype_ : InferDataType(X);

  if (dtype_ == TensorProto_DataType_UNDEFINED)
    return LOTUS_MAKE_STATUS(LOTUS, FAIL,
                             "Could not infer data type from input tensor with data type ",
                             X.DataType());
  status = RandomUniformCompute(low_, high_, generator_, dtype, *Y);

  return status;
}

/* 
alternative interpretation of the spec is that the input tensor contains the dimensions as ints.
Keeping this temporarily in case we go back to that.

// read shape information from input tensor and create output tensor with it
static Status CreateOutputTensorFromTensorValues(OpKernelContext* ctx, const Tensor& X, Tensor** Y) {
  const TensorShape& shape = X.Shape();
  auto size = shape.Size();
  auto num_dims = shape.NumDimensions();

  if (num_dims != 1) {
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "Expected 1 dimension tensor with shape information. Dimensions=", num_dims);
  }

  std::vector<int64_t> dims;
  dims.reserve(shape.Size());

  auto data = gsl::make_span(tensor.Data<int64_t>(), shape.Size());
  dims.insert(dims.cbegin(), data.cbegin(), data.cend());

  *Y = ctx->Output(0, TensorShape(dims));

  return Status::OK();
}
*/

// create output tensor using shape of input tensor
static Status CreateOutputTensorFromTensorShape(OpKernelContext* ctx, const Tensor& X, Tensor** Y) {
  const TensorShape& shape = X.Shape();

  *Y = ctx->Output(0, shape);

  return Status::OK();
}

static TensorProto::DataType InferDataType(const Tensor& tensor) {
  auto tensor_type = tensor.DataType();
  TensorProto::DataType dtype = TensorProto_DataType_UNDEFINED;

  if (tensor_type == DataTypeImpl::GetType<float>())
    dtype = TensorProto_DataType_FLOAT;
  else if (tensor_type == DataTypeImpl::GetType<double>())
    dtype = TensorProto_DataType_DOUBLE;
  else {
    // unsupported. return UNDEFINED
  }

  return dtype;
}

static Status RandomNormalCompute(float mean, float scale,
                                  std::default_random_engine generator,
                                  TensorProto::DataType dtype, Tensor& Y) {
  switch (dtype) {
    case TensorProto::FLOAT: {
      GenerateData<float, std::normal_distribution<float>>(
          generator, std::normal_distribution<float>{mean, scale}, Y);
      break;
    }
    case TensorProto::FLOAT16: {
      LOTUS_NOT_IMPLEMENTED("FLOAT16 is not supported");
    }
    case TensorProto::DOUBLE: {
      GenerateData<double, std::normal_distribution<double>>(
          generator, std::normal_distribution<double>{mean, scale}, Y);
      break;
    }
    default:
      LOTUS_THROW("Invalid data type of ", dtype);
  }

  return Status::OK();
}

static Status RandomUniformCompute(float low, float high,
                                   std::default_random_engine generator,
                                   TensorProto::DataType dtype,
                                   Tensor& Y) {
  switch (dtype) {
    case TensorProto::FLOAT: {
      GenerateData<float, std::uniform_real_distribution<float>>(
          generator, std::uniform_real_distribution<float>{low, high}, Y);
      break;
    }
    case TensorProto::FLOAT16: {
      LOTUS_NOT_IMPLEMENTED("FLOAT16 is not supported");
    }
    case TensorProto::DOUBLE: {
      GenerateData<double, std::uniform_real_distribution<double>>(
          generator, std::uniform_real_distribution<double>{low, high}, Y);
      break;
    }
    default:
      LOTUS_THROW("Invalid data type of ", dtype);
  }

  return Status::OK();
}

template <typename T, typename TDistribution>
void GenerateData(std::default_random_engine generator, TDistribution distribution, Tensor& tensor) {
  auto out = gsl::make_span(tensor.MutableData<T>(), tensor.Shape().Size());

  std::for_each(out.begin(), out.end(), [&generator, &distribution](T& value) { value = distribution(generator); });
}

}  // namespace Lotus

#include "core/providers/cpu/ml/onehotencoder.h"
/**
https://github.com/onnx/onnx/blob/master/onnx/defs/traditionalml/defs.cc
ONNX_OPERATOR_SCHEMA(OneHotEncoder)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
Replace the inputs with an array of ones and zeros, where the only
one is the zero-based category that was passed in.  The total category count
will determine the length of the vector. For example if we pass a
tensor with a single value of 4, and a category count of 8, the
output will be a tensor with 0,0,0,0,1,0,0,0 .

This operator assumes every input in X is of the same category set
(meaning there is only one category count).

If the input is a tensor of float, int32, or double, the data will be cast
to int64s and the cats_int64s category list will be used for the lookups.
)DOC")
.Input(0, "X", "Data to be encoded", "T")
.Output(0, "Y", "encoded output data", "tensor(float)")
.TypeConstraint("T", { "tensor(string)", "tensor(int64)","tensor(int32)", "tensor(float)","tensor(double)" }, " allowed types.")
.Attr("cats_int64s", "list of categories, ints", AttributeProto::INTS, OPTIONAL)
.Attr("cats_strings", "list of categories, strings", AttributeProto::STRINGS, OPTIONAL)
.Attr(
"zeros",
"if true and category is not present, will return all zeros, if false and missing category, operator will return false",
AttributeProto::INT,
OPTIONAL);
*/

namespace Lotus {
namespace ML {

REGISTER_KERNEL(KernelDefBuilder("OneHotEncoder")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", {DataTypeImpl::GetTensorType<int64_t>()}),
                OneHotEncoderOp<int64_t>);

REGISTER_KERNEL(KernelDefBuilder("OneHotEncoder")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>()}),
                OneHotEncoderOp<float>);

REGISTER_KERNEL(KernelDefBuilder("OneHotEncoder")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", {DataTypeImpl::GetTensorType<double>()}),
                OneHotEncoderOp<double>);

REGISTER_KERNEL(KernelDefBuilder("OneHotEncoder")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", {DataTypeImpl::GetTensorType<std::string>()}),
                OneHotEncoderOp<std::string>);

/*
REGISTER_KERNEL(KernelDefBuilder("OneHotEncoder")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", { DataTypeImpl::GetTensorType<int>() }),
                OneHotEncoderOp<int>);
                

REGISTER_KERNEL(KernelDefBuilder("OneHotEncoder")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", { DataTypeImpl::GetTensorType<double>() }),
                OneHotEncoderOp<double>);
*/

template <typename T>
OneHotEncoderOp<T>::OneHotEncoderOp(const OpKernelInfo& info) : OpKernel(info), zeros_(1), num_categories_(0) {
  std::vector<int64_t> tmp_cats_int64s;
  std::vector<std::string> tmp_cats_strings;
  info.GetAttrs<int64_t>("cats_int64s", tmp_cats_int64s);   // optional
  info.GetAttrs<string>("cats_strings", tmp_cats_strings);  // optional
  info.GetAttr<int64_t>("zeros", &zeros_);                  // optional
  LOTUS_ENFORCE(tmp_cats_int64s.empty() || tmp_cats_strings.empty());
  if (!tmp_cats_int64s.empty()) {
    num_categories_ = tmp_cats_int64s.size();
    for (size_t idx = 0, end = tmp_cats_int64s.size(); idx < end; ++idx) {
      cats_int64s_[tmp_cats_int64s[idx]] = idx;
    }
  } else {
    num_categories_ = tmp_cats_strings.size();
    for (size_t idx = 0, end = tmp_cats_strings.size(); idx < end; ++idx) {
      cats_strings_[tmp_cats_strings[idx]] = idx;
    }
  }
  LOTUS_ENFORCE(num_categories_ > 0);
}

template <typename T>
Common::Status OneHotEncoderOp<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& input_shape = X->Shape();
  LOTUS_ENFORCE(input_shape.NumDimensions() <= 2);

  std::vector<int64_t> output_shape(input_shape.GetDims());
  output_shape.push_back(num_categories_);

  Tensor* Y = context->Output(0, TensorShape(output_shape));
  auto y_data = Y->MutableData<float>();
  std::fill_n(y_data, Y->Shape().Size(), 0.0f);

  auto x_data = X->Data<T>();
  std::unordered_map<int64_t, size_t>::const_iterator idx;
  for (int64_t i = 0; i < input_shape.Size(); ++i) {
    auto int_idx = cats_int64s_.find(static_cast<int64_t>(x_data[i]));
    if (int_idx != cats_int64s_.cend())
      y_data[i * num_categories_ + int_idx->second] = 1.0f;
    else if (!zeros_)
      return Status(LOTUS, FAIL, "Unknown Category and zeros = 0.");
  }
  return Status::OK();
}

template <>
Common::Status OneHotEncoderOp<std::string>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& input_shape = X->Shape();
  LOTUS_ENFORCE(input_shape.NumDimensions() <= 2);

  std::vector<int64_t> output_shape(input_shape.GetDims());
  output_shape.push_back(num_categories_);

  Tensor* Y = context->Output(0, TensorShape(output_shape));
  auto y_data = Y->MutableData<float>();
  std::fill_n(y_data, Y->Shape().Size(), 0.0f);

  auto x_data = X->Data<std::string>();
  for (int64_t i = 0; i < input_shape.Size(); ++i) {
    auto str_idx = cats_strings_.find(x_data[i]);
    if (str_idx != cats_strings_.cend())
      y_data[i * num_categories_ + str_idx->second] = 1.0f;
    else if (!zeros_)
      return Status(LOTUS, FAIL, "Unknown Category and zeros = 0.");
  }
  return Status::OK();
}

}  // namespace ML
}  // namespace Lotus

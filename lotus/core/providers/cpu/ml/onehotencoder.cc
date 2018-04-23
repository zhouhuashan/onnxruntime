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
	          				.TypeConstraint("T", { DataTypeImpl::GetTensorType<int64_t>() }),
				        OneHotEncoderOp<int64_t>);

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

REGISTER_KERNEL(KernelDefBuilder("OneHotEncoder")
  .Domain(LotusIR::kMLDomain)
  .SinceVersion(1)
  .Provider(LotusIR::kCpuExecutionProvider)
  .TypeConstraint("T", { DataTypeImpl::GetTensorType<std::string>() }),
  OneHotEncoderOp<std::string>);
*/

template <typename T>
OneHotEncoderOp<T>::OneHotEncoderOp(const OpKernelInfo& info) : OpKernel(info), zeros_(1), category_(0) {
  info.GetAttrs<int64_t>("cats_int64s", cats_int64s_);    // optional
  info.GetAttrs<string>("cats_strings", cats_strings_);    // optional
  info.GetAttr<int64_t>("zeros", &zeros_);  // optional

  LOTUS_ENFORCE(cats_int64s_.empty() || cats_strings_.empty());
  if (!cats_int64s_.empty()) category_ = cats_int64s_.size();
  else category_ = cats_strings_.size();
  LOTUS_ENFORCE(category_ > 0);
}

template<typename T>
std::vector<int64_t> OneHotEncoderOp<T>::InferOutputSize(const TensorShape& input_shape) const
{  
  return std::vector<int64_t> ({ input_shape[0], input_shape[1], category_ });
}



template <typename T>
Common::Status OneHotEncoderOp<T>::Compute(OpKernelContext* context) const {	
  const Tensor* X = context->Input<Tensor>(0);  
  const TensorShape& input_shape = X->Shape();
  LOTUS_ENFORCE(input_shape.NumDimensions() == 2);
  
  std::vector<int64_t> output_shape = InferOutputSize(input_shape);
  Tensor* Y = context->Output(0, TensorShape(output_shape));
  
  if (context->InputType(0) == DataTypeImpl::GetType<std::string>())
  {
    // TODO: Handle String type
  }
  else
  {    
    auto* x_data = X->Data<T>();
    float* y_data = Y->MutableData<float>();    
    int64_t count = 0;
    
    for (int64_t i = 0; i < input_shape.Size(); ++i)
    {
      bool found = false;
      for (int64_t j = 0; j < category_; ++j)
        if (static_cast<int64_t>(x_data[i]) == cats_int64s_[j])
        {
          y_data[count++] = 1.0f;
          found = true;
        }
        else
        {
          y_data[count++] = 0.0f;
        }

      if (!found && !zeros_)
      {
        return Status(LOTUS, FAIL, "Unknown Category and zeros = 0..");
      }
    }
  }
  return Status::OK();
}

}  // namespace ML
}  // namespace Lotus

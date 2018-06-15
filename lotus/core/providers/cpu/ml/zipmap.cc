#include "core/providers/cpu/ml/zipmap.h"

/**
https://github.com/onnx/onnx/blob/master/onnx/defs/traditionalml/defs.cc
ONNX_OPERATOR_SCHEMA(ZipMap)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
Makes a map from the input and the attributes.
Assumes input 0 are the values, and the keys are specified by the attributes.
Must provide keys in either classlabels_strings or classlabels_int64s (but not both).
Input 0 may have a batch size larger than 1,
but each input in the batch must be the size of the keys specified by the attributes.
The order of the input and attributes determines the key-value mapping.
)DOC")
.Input(0, "X", "The input values", "tensor(float)")
.Output(0, "Z", "The output map", "T")
.TypeConstraint(
"T",
{ "seq(map(string, float))", "seq(map(int64, float))" },
" allowed types.")
.Attr("classlabels_strings", "keys if using string keys", AttributeProto::STRINGS, OPTIONAL)
.Attr("classlabels_int64s", "keys if using int keys", AttributeProto::INTS, OPTIONAL);
*/

namespace Lotus {
namespace ML {
REGISTER_KERNEL(KernelDefBuilder("ZipMap")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T", {DataTypeImpl::GetType<std::vector<std::map<std::string, float>>>(),
                                          DataTypeImpl::GetType<std::vector<std::map<std::int64_t, float>>>()}),
                ZipMapOp);

ZipMapOp::ZipMapOp(const OpKernelInfo& info) : OpKernel(info) {
  op_kernel_info_.GetAttrs<std::string>("classlabels_strings", classlabels_strings_);  // optional
  op_kernel_info_.GetAttrs<int64_t>("classlabels_int64s", classlabels_int64s_);        // optional
  LOTUS_ENFORCE(classlabels_strings_.empty() ^ classlabels_int64s_.empty(),
                "Must provide classlabels_strings or classlabels_int64s but not both.");
  using_strings_ = !classlabels_strings_.empty();
}

Common::Status ZipMapOp::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  const vector<int64_t> x_dims = x_shape.GetDims();

  if (x_dims.empty())
  {
    return Status(LOTUS,
                  INVALID_ARGUMENT,
                  "Zipmap does not support empty dim count");
  }

  int64_t batch_size = x_dims.size() > 1 ? x_dims[0] : 1;
  int64_t features_per_Batch = x_dims[x_dims.size() - 1];

  if (x_dims.size() > 2)
  {
    for (size_t dim = 1; dim < x_dims.size() - 1; dim++)
    {
      if (x_dims[dim] != 1)
      {
        return Status(LOTUS,
                      INVALID_ARGUMENT,
                      "Zipmap only supports inputs with 1 or 2 dims that are not size of 1.");
      }
    }
  }

  const float* x_data = X.Data<float>();

  if (using_strings_) {
    if (features_per_Batch != static_cast<int64>(classlabels_strings_.size())) {
      return Status(LOTUS,
                    INVALID_ARGUMENT,
                    "Input features_per_Batch[" + std::to_string(features_per_Batch) +
                        "] != number of classlabels[" + std::to_string(classlabels_strings_.size()) + "]");
    }
    auto* y_data = context->Output<std::vector<std::map<std::string, float>>>(0);
    //auto* y_data = Y->MutableData<std::vector<std::map<std::string, float>>>();
    y_data->resize(batch_size);
    int64_t current_weight_0 = 0;
    for (int n = 0; n < batch_size; n++) {
      std::map<std::string, float> map1;
      for (int j = 0; j < features_per_Batch; j++) {
        map1[classlabels_strings_[j]] = x_data[current_weight_0 + j];
      }
      current_weight_0 += features_per_Batch;
      (*y_data)[n] = map1;
    }
  } else {
    if (features_per_Batch != static_cast<int64>(classlabels_int64s_.size())) {
      return Status(LOTUS,
                    INVALID_ARGUMENT,
                    "Input features_per_Batch[" + std::to_string(features_per_Batch) +
                        "] != number of classlabels[" + std::to_string(classlabels_int64s_.size()) + "]");
    }
    auto* y_data = context->Output<std::vector<std::map<std::int64_t, float>>>(0);
    //auto* y_data = Y->MutableData<std::vector<std::map<int64_t, float>>>();
    y_data->resize(batch_size);
    int64_t current_weight_0 = 0;
    for (int n = 0; n < batch_size; n++) {
      std::map<int64_t, float> map2;
      for (int j = 0; j < features_per_Batch; j++) {
        map2[classlabels_int64s_[j]] = x_data[current_weight_0 + j];
      }
      current_weight_0 += features_per_Batch;
      (*y_data)[n] = map2;
    }
  }
  return Common::Status::OK();
}
}  // namespace ML
}  // namespace Lotus

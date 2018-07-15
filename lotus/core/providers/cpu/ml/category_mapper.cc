#include "core/providers/cpu/ml/category_mapper.h"
#include <algorithm>
#include <gsl/span>
using namespace Lotus::Common;

namespace Lotus {
namespace ML {

REGISTER_KERNEL(KernelDefBuilder("CategoryMapper")
                    .Domain(LotusIR::kMLDomain)
                    .SinceVersion(1)
                    .Provider(LotusIR::kCpuExecutionProvider)
                    .TypeConstraint("T1",
                                    std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>(),
                                                            DataTypeImpl::GetTensorType<int64_t>()})
                    .TypeConstraint("T2",
                                    std::vector<MLDataType>{DataTypeImpl::GetTensorType<std::string>(),
                                                            DataTypeImpl::GetTensorType<int64_t>()}),
                CategoryMapper);

Status CategoryMapper::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& shape = X.Shape();
  Tensor& Y = *context->Output(0, TensorShape(shape));

  auto input_type = X.DataType();

  if (input_type == DataTypeImpl::GetType<std::string>()) {
    if (Y.DataType() != DataTypeImpl::GetType<int64_t>())
      return Status(LOTUS, FAIL, "Input of string must have output of int64");

    auto input = gsl::make_span(X.Data<std::string>(), shape.Size());
    auto output = gsl::make_span(Y.MutableData<int64_t>(), shape.Size());
    auto out = output.begin();

    // map isn't going to change so get end() once instead of calling inside the for_each loop
    const auto map_end = string_to_int_map_.end();

    std::for_each(input.cbegin(), input.cend(),
                  [&out, &map_end, this](const std::string& value) {
                    auto map_to = string_to_int_map_.find(value);
                    *out = map_to == map_end ? default_int_ : map_to->second;
                    ++out;
                  });
  } else {
    if (Y.DataType() != DataTypeImpl::GetType<std::string>())
      return Status(LOTUS, FAIL, "Input of int64 must have output of string ");

    auto input = gsl::make_span(X.Data<int64_t>(), shape.Size());
    auto output = gsl::make_span(Y.MutableData<std::string>(), shape.Size());
    auto out = output.begin();

    const auto map_end = int_to_string_map_.end();

    std::for_each(input.cbegin(), input.cend(),
                  [&out, &map_end, this](const int64_t& value) {
                    auto map_to = int_to_string_map_.find(value);
                    *out = map_to == map_end ? default_string_ : map_to->second;
                    ++out;
                  });
  }

  return Status::OK();
}

}  // namespace ML
}  // namespace Lotus

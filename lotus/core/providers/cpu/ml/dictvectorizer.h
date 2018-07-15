#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace Lotus {
namespace ML {

class DictVectorizerOp final : public OpKernel {
 public:
  DictVectorizerOp(const OpKernelInfo& info) : OpKernel(info) {
    op_kernel_info_.GetAttrs<std::string>("string_vocabulary", string_index_);
    op_kernel_info_.GetAttrs<int64_t>("int64_vocabulary", int_index_);
    LOTUS_ENFORCE(string_index_.empty() ^ int_index_.empty(),
                  "Must provide string_vocabulary or int64_vocabulary but not both.");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename TKey, typename TVal>
  Status ComputeWithType(OpKernelContext* ctx,
                         const std::vector<TKey>& vocabulary,
                         TVal default_value) const {
    auto map = ctx->Input<std::map<TKey, TVal> >(0);
    std::vector<int64_t> dims{1, static_cast<int64_t>(vocabulary.size())};
    auto Y = ctx->Output(0, TensorShape(dims));
    auto* y_data = Y->MutableData<TVal>();
    int64_t write_index = 0;
    //for each vocab word, if its in the input, use that value, otherwise output 0f
    for (int64_t i = 0, end = static_cast<int64_t>(vocabulary.size()); i < end; ++i) {
      auto index = map->find(vocabulary[i]);
      if (index != map->end()) {
        y_data[write_index] = index->second;
      } else {
        y_data[write_index] = default_value;
      }
      write_index++;
    }
    return Status::OK();
  }

  std::vector<std::string> string_index_;
  std::vector<int64_t> int_index_;
};
}  // namespace ML

}  // namespace Lotus

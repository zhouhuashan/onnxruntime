// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class DynamicSliceBase
{
protected:
  using DIMS = std::vector<int64_t>;

  struct Prepare {
    const uint8_t*        input_base;
    const std::string*    input_str_base;
    uint8_t*              output_base;
    std::string*          output_str_base;
    uint64_t              bytes_to_copy;
    uint64_t              element_bytes;
    uint64_t              element_to_copy;
    DIMS                  element_offsets;

    Prepare(): input_base      (nullptr),
               input_str_base  (nullptr),
               output_base     (nullptr),
               output_str_base (nullptr),
               bytes_to_copy   (0),
               element_bytes   (0),
               element_to_copy (0) {}
  }; // struct Prepare

  template<typename Tind>
  Status PrepareForCompute(OpKernelContext* context, Prepare& p) const;

private:

  template<typename Tind>
  void FindAllOffset(DIMS&, const DIMS&, const DIMS&, int64_t) const;
  template<typename Tind>
  int64_t AdjustOutputShape(DIMS&, const DIMS&, const DIMS&, const Tensor*, std::string&) const;
  mutable DIMS merged_starts;
}; // class DynamicSliceBase 

class DynamicSlice final : public OpKernel, protected DynamicSliceBase {
public:
  explicit DynamicSlice(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
private:
  Status SliceNumber(const Prepare& p) const;
  Status SliceString(const Prepare& p) const;
};

}



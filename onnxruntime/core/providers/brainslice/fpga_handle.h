// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
#include "core/providers/brainslice/fpga_util.h"
#include "core/providers/provider_factories.h"

namespace onnxruntime {
namespace fpga {

constexpr int msg_header_bytes = 1024;

class FPGAHandle {
 public:
  FPGAHandle(FPGAInfo info);
  virtual ~FPGAHandle() {}

  const BS_Capabilities& GetCapacities() const {
    return capacities_;
  }

  Status LoadMatrix(const std::vector<half_float::half>& matrix, const int rows, const int cols,
                    const int matix_addr, const bool row_major, const ISA_Mem mem_type) const;

  Status LoadVector(const std::vector<half_float::half>& vector, const int vec_addr, const ISA_Mem mem_type) const;

  Status SendSync(std::function<int32_t(void*, size_t*)> prepare_request, std::function<int32_t(void*, size_t)> process_response) const;

 private:
  // To load a firmware to brainslice, three files are needed:
  // 1. the instructions bin file
  // 2. the data bin file
  // 3. the schema bin file
  Status LoadFirmware(const std::string& inst_file,
                      const std::string& data_file,
                      const std::string& schema_file);

  Status LoadFirmware(std::vector<uint32_t>&& inst,
                      std::vector<uint32_t>&& data,
                      std::vector<uint64_t>&& schema);

  uint32_t ip_;

  BS_Capabilities capacities_;

  size_t max_request_size_;
};
}  // namespace fpga
}  // namespace onnxruntime

#pragma once

#include "core/common/status.h"
#include "core/platform/env.h"
#include "core/framework/customregistry.h"
#include "core/framework/custom_ops_author.h"
#include "core/common/common.h"
#include <string>
#include <memory>

namespace Lotus {
class CustomOpsLoader final {
 public:
  CustomOpsLoader() = default;
  Common::Status LoadCustomOps(const std::vector<std::string>& dso_list,
                               std::shared_ptr<CustomRegistry>& custom_registry);
  ~CustomOpsLoader();

 private:
  const std::string kGetAllKernelsSymbol = "GetAllKernels";
  const std::string kGetAllSchemasSymbol = "GetAllSchemas";
  const std::string kFreeKernelsContainerSymbol = "FreeKernelsContainer";
  const std::string kFreeSchemasContainerSymbol = "FreeSchemasContainer";

  struct DsoData {
    void* lib_handle = nullptr;
    KernelsContainer* kernels_container = nullptr;
    SchemasContainer* schemas_container = nullptr;
  };
  std::map<std::string, DsoData> dso_name_data_map_;

  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(CustomOpsLoader);
};
}  // namespace Lotus

#include "core/framework/environment.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"

namespace Lotus {

using namespace Lotus::Common;

Status Environment::Initialize() {
  auto status = Status::OK();

  try {
    // Register Microsoft domain with min/max op_set version as 1/1.
    onnx::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(LotusIR::kMSDomain, 1, 1);

    // Register Microsoft domain ops.
    status = LotusIR::MsOpRegistry::RegisterMsOps();
    LOTUS_RETURN_IF_ERROR(status);

    // TODO: Should register microsoft domain kernels here.

    // LotusDeviceManager
    status = AllocatorManager::Create(allocation_manager_);
  } catch (std::exception& ex) {
    status = Status{LOTUS, StatusCode::RUNTIME_EXCEPTION, std::string{"Exception caught: "} + ex.what()};
  } catch (...) {
    status = Status{LOTUS, StatusCode::RUNTIME_EXCEPTION};
  }

  return status;
}

}  // namespace Lotus

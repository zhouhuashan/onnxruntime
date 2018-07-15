#include "core/framework/environment.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"

namespace Lotus {

using namespace Lotus::Common;

std::once_flag schemaRegistrationOnceFlag;

Status Environment::Initialize() {
  auto status = Status::OK();

  try {
    // Register Microsoft domain with min/max op_set version as 1/1.
    std::call_once(schemaRegistrationOnceFlag, []() {
      onnx::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(LotusIR::kMSDomain, 1, 1);
    });

    // Register Microsoft domain ops.
    status = LotusIR::MsOpRegistry::RegisterMsOps();
    LOTUS_RETURN_IF_ERROR(status);

    // TODO: Should register microsoft domain kernels here.

  } catch (std::exception& ex) {
    status = Status{LOTUS, Common::RUNTIME_EXCEPTION, std::string{"Exception caught: "} + ex.what()};
  } catch (...) {
    status = Status{LOTUS, Common::RUNTIME_EXCEPTION};
  }

  return status;
}

Environment::~Environment() {
  ::google::protobuf::ShutdownProtobufLibrary();
}

}  // namespace Lotus

#include "core/framework/environment.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"

namespace Lotus {

using namespace Lotus::Common;
using namespace onnx;

std::once_flag schemaRegistrationOnceFlag;

Status Environment::Initialize() {
  auto status = Status::OK();

  try {
    // Register Microsoft domain with min/max op_set version as 1/1.
    std::call_once(schemaRegistrationOnceFlag, []() {
      onnx::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(LotusIR::kMSDomain, 1, 1);
    });

    // Register MemCpy schema;

    // These ops are internal-only, so register outside of onnx
    ONNX_OPERATOR_SCHEMA(MemcpyFromHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

    ONNX_OPERATOR_SCHEMA(MemcpyToHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

    // Register Microsoft domain ops.
    status = LotusIR::MsOpRegistry::RegisterMsOps();
    LOTUS_RETURN_IF_ERROR(status);

    // Register MemCpy schema;

    // These ops are internal-only, so register outside of onnx
    ONNX_OPERATOR_SCHEMA(MemcpyFromHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
                Internal copy node
                )DOC");

    ONNX_OPERATOR_SCHEMA(MemcpyToHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
                Internal copy node
                )DOC");

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

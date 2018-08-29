#include "core/framework/environment.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "contrib_ops/contrib_ops.h"

#ifndef LOTUS_HAVE_ATTRIBUTE
#ifdef __has_attribute
#define LOTUS_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define LOTUS_HAVE_ATTRIBUTE(x) 0
#endif
#endif

// LOTUS_ATTRIBUTE_UNUSED
//
// Prevents the compiler from complaining about or optimizing away variables
// that appear unused.
#if LOTUS_HAVE_ATTRIBUTE(unused) || (defined(__GNUC__) && !defined(__clang__))
#undef LOTUS_ATTRIBUTE_UNUSED
#define LOTUS_ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
#define LOTUS_ATTRIBUTE_UNUSED
#endif
namespace Lotus {

using namespace ::Lotus::Common;
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
    LOTUS_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MemcpyFromHost)
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

    LOTUS_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MemcpyToHost)
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

    // Register contributed schemas.
    // The corresponding kernels are registered inside the appropriate execution provider.
    ML::RegisterContribSchemas();
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

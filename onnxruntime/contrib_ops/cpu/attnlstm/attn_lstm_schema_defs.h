#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "onnx/defs/schema.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace ml {

::ONNX_NAMESPACE::OpSchema& RegisterAttnLSTMContribOpSchema(::ONNX_NAMESPACE::OpSchema&& op_schema);

}
}
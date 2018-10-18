#pragma once

#include "core/common/status.h"
#include "core/framework/error_code.h"

namespace onnxruntime {
ONNXStatusPtr ToONNXStatus(const onnxruntime::common::Status& st);
};
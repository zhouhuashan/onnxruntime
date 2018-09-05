#pragma once

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/environment.h"
#include "core/framework/ml_value.h"

using namespace std;
namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace Lotus;

int OnnxRuntimeTensorToNumpyType(const MLDataType& tensor_type);

void CreateGenericMLValue(AllocatorPtr alloc, py::object& value, MLValue* p_mlvalue);

}
}

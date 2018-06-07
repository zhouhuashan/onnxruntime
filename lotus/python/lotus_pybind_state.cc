#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL lotus_python_ARRAY_API
#include <numpy/arrayobject.h>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/allocatormgr.h"
#include <core/framework/environment.h>
#include "core/framework/inference_session.h"
#include "core/graph/graph.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4267 4996 4503 4003)
#endif  // _MSC_VER

#include <iterator>

namespace lotus {
namespace python {

namespace py = pybind11;
using namespace Lotus;
using namespace Lotus::Logging;

int LotusToNumpyType(const MLDataType& lotus_type) {
  static std::map<MLDataType, int> type_map{
      {DataTypeImpl::GetType<bool>(), NPY_BOOL},
      {DataTypeImpl::GetType<float>(), NPY_FLOAT},
      {DataTypeImpl::GetType<double>(), NPY_DOUBLE},
      {DataTypeImpl::GetType<int32_t>(), NPY_INT},
      {DataTypeImpl::GetType<int8_t>(), NPY_INT8},
      {DataTypeImpl::GetType<uint8_t>(), NPY_UINT8},
      {DataTypeImpl::GetType<int16_t>(), NPY_INT16},
      {DataTypeImpl::GetType<uint16_t>(), NPY_UINT16},
      {DataTypeImpl::GetType<int64_t>(), NPY_LONGLONG},
      {DataTypeImpl::GetType<uint64_t>(), NPY_ULONGLONG},
      {DataTypeImpl::GetType<std::string>(), NPY_STRING},
  };

  const auto it = type_map.find(lotus_type);
  if (it == type_map.end()) {
    throw std::runtime_error("No corresponding Numpy type for MLDataType.");
  } else {
    return it->second;
  }
}

const MLDataType& NumpyToLotusType(int numpy_type) {
  static std::map<int, MLDataType> type_map{
      {NPY_BOOL, DataTypeImpl::GetType<bool>()},
      {NPY_FLOAT, DataTypeImpl::GetType<float>()},
      {NPY_DOUBLE, DataTypeImpl::GetType<double>()},
      {NPY_INT, DataTypeImpl::GetType<int32_t>()},
      {NPY_INT8, DataTypeImpl::GetType<int8_t>()},
      {NPY_UINT8, DataTypeImpl::GetType<uint8_t>()},
      {NPY_INT16, DataTypeImpl::GetType<int16_t>()},
      {NPY_UINT16, DataTypeImpl::GetType<uint16_t>()},
      {NPY_LONG,
       sizeof(long) == sizeof(int) ? DataTypeImpl::GetType<int32_t>()
                                   : DataTypeImpl::GetType<int64_t>()},
      {NPY_LONGLONG, DataTypeImpl::GetType<int64_t>()},
      {NPY_ULONGLONG, DataTypeImpl::GetType<uint64_t>()},
      {NPY_STRING, DataTypeImpl::GetType<std::string>()}};

  const auto it = type_map.find(numpy_type);
  if (it == type_map.end()) {
    throw std::runtime_error("Numpy_type " + std::to_string(numpy_type) +
                             " can't be converted to MLDataType");
  } else {
    return it->second;
  }
}

static AllocatorPtr& GetAllocator() {
  static AllocatorPtr alloc = std::make_shared<CPUAllocator>();
  return alloc;
}

static const SessionOptions& GetDefaultCPUSessionOptions() {
  static SessionOptions so;
  return so;
}

void CreateTensorMLValue(AllocatorPtr alloc,
                         PyArrayObject* pyObject, MLValue* p_mlvalue) {
  PyArrayObject* darray = PyArray_GETCONTIGUOUS(pyObject);
  bool dref = false;
  try {
    const auto npy_type = PyArray_TYPE(darray);

    // numpy requires long int as its dims.
    int ndim = PyArray_NDIM(darray);
    npy_intp* npy_dims = PyArray_DIMS(darray);
    std::vector<int64_t> dims;
    for (int i = 0; i < ndim; ++i) {
      dims.push_back(npy_dims[i]);
    }

    TensorShape shape(dims);
    auto element_type = NumpyToLotusType(npy_type);
    void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
    memcpy(buffer, static_cast<void*>(PyArray_DATA(darray)), element_type->Size() * shape.Size());

    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                                shape,
                                                                static_cast<void*>(buffer),
                                                                alloc->Info(), alloc);
    p_mlvalue->Init(p_tensor.release(),
                    DataTypeImpl::GetType<Tensor>(),
                    DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  } catch (...) {
    if (!dref) {
      Py_XDECREF(darray);
      dref = true;
    }

    // allocator should be able to gc the memory created by it.
    // ...
  }

  if (!dref) {
    Py_XDECREF(darray);
  }
}

class SessionObjectInitializer {
 public:
  typedef const SessionOptions& Arg1;
  typedef Logging::LoggingManager* Arg2;
  operator Arg1() {
    return GetDefaultCPUSessionOptions();
  }

  operator Arg2() {
    static std::string default_logger_id{"Default"};
    static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CLogSink{}},
                                                  Severity::kWARNING, false, LoggingManager::InstanceType::Default,
                                                  &default_logger_id};
    return &default_logging_manager;
  }

  static SessionObjectInitializer Get() {
    return SessionObjectInitializer();
  }
};

void addGlobalMethods(py::module& m) {
  m.def("get_session_initializer", &SessionObjectInitializer::Get, "Return a default session object initializer.");
}

void addObjectMethods(py::module& m) {
  py::class_<LotusIR::NodeArg>(m, "NodeArg")
      .def_property_readonly("name", &LotusIR::NodeArg::Name)
      .def_property_readonly("type", [](const LotusIR::NodeArg& na) -> std::string {
        return *(na.Type());
      })
      .def_property_readonly("shape", [](const LotusIR::NodeArg& na) -> std::vector<py::object> {
        auto shape = na.Shape();
        std::vector<py::object> arr;
        if (shape == nullptr || shape->dim_size() == 0) {
          return arr;
        }

        arr.resize(shape->dim_size());
        for (int i = 0; i < shape->dim_size(); ++i) {
          if (shape->dim(i).has_dim_value()) {
            arr[i] = py::cast(shape->dim(i).dim_value());
          } else if (shape->dim(i).has_dim_param()) {
            arr[i] = py::none();
          }
        }
        return arr;
      });

  py::class_<SessionObjectInitializer>(m, "SessionObjectInitializer");
  py::class_<InferenceSession>(m, "InferenceSession")
      .def(py::init<SessionObjectInitializer, SessionObjectInitializer>())
      .def("load_model", [](InferenceSession* rt, const std::string& path) {
        auto status = rt->Load(path);

        if (!status.IsOK()) {
          throw std::runtime_error(status.ToString().c_str());
        }

        status = rt->Initialize();
        if (!status.IsOK()) {
          throw std::runtime_error(status.ToString().c_str());
        }
      })
      .def("run", [](InferenceSession* sess, std::vector<std::string> output_names, std::map<std::string, py::object> feed) -> std::vector<py::object> {
        NameMLValMap feeds;
        for (auto _ : feed) {
          MLValue ml_value;
          PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(_.second.ptr());
          CreateTensorMLValue(GetAllocator(), arr, &ml_value);
          feeds.insert(std::make_pair(_.first, ml_value));
        }

        std::vector<MLValue> fetcher;
        // TODO: expose run_options and session options to python
        RunOptions run_options;

        Common::Status status = sess->Run(run_options, feeds, output_names, &fetcher);
        if (!status.IsOK()) {
          throw std::runtime_error(status.ToString().c_str());
        }

        std::vector<py::object> rfetch;
        rfetch.reserve(fetcher.size());

        for (auto _ : fetcher) {
          const Tensor& rtensor = _.Get<Tensor>();
          std::vector<npy_intp> npy_dims;
          const TensorShape& shape = rtensor.Shape();

          for (size_t n = 0; n < shape.NumDimensions(); ++n) {
            npy_dims.push_back(shape[n]);
          }

          MLDataType dtype = rtensor.DataType();
          py::object obj = py::reinterpret_steal<py::object>(PyArray_SimpleNew(
              shape.NumDimensions(), npy_dims.data(), LotusToNumpyType(dtype)));
          void* outPtr = static_cast<void*>(
              PyArray_DATA(reinterpret_cast<PyArrayObject*>(obj.ptr())));
          memcpy(outPtr, rtensor.DataRaw(dtype), dtype->Size() * shape.Size());
          rfetch.push_back(obj);
        }
        return rfetch;
      })
      .def_property_readonly("inputs_meta", [](const InferenceSession* sess) -> const std::vector<const LotusIR::NodeArg*>& {
        auto res = sess->GetInputs();
        if (!res.first.IsOK()) {
          throw std::runtime_error(res.first.ToString().c_str());
        } else {
          return *(res.second);
        }
      })
      .def_property_readonly("outputs_meta", [](const InferenceSession* sess) -> const std::vector<const LotusIR::NodeArg*>& {
        auto res = sess->GetOutputs();
        if (!res.first.IsOK()) {
          throw std::runtime_error(res.first.ToString().c_str());
        } else {
          return *(res.second);
        }
      });
}  //  end of addLotusObjectMethods

PYBIND11_MODULE(lotus_pybind11_state, m) {
  m.doc() = "pybind11 stateful interface to lotus runtime";

  auto initialize = [&]() {
    // Initialization of the module
    ([]() -> void {
      // import_array1() forces a void return value.
      import_array1();
    })();

    static std::unique_ptr<Environment> env;
    auto status = Environment::Create(env);
    if (!status.IsOK()) {
      throw std::runtime_error(status.ToString().c_str());
    }

    static bool initialized = false;
    if (initialized) {
      return;
    }
    initialized = true;
  };
  initialize();

  addGlobalMethods(m);
  addObjectMethods(m);
}

}  // namespace python
}  // namespace lotus

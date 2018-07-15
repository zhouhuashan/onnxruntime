#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL lotus_python_ARRAY_API
#include <numpy/arrayobject.h>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/framework/allocatormgr.h"
#include <core/framework/environment.h>
#include "core/framework/inference_session.h"
#include "core/graph/graph.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4267 4996 4503 4003)
#endif  // _MSC_VER

#include <iterator>
using namespace std;
namespace lotus {
namespace python {

namespace py = pybind11;
using namespace Lotus;
using namespace Lotus::Logging;

int LotusTensorToNumpyType(const MLDataType& tensor_type) {
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
      {DataTypeImpl::GetType<std::string>(), NPY_OBJECT},
  };

  const auto it = type_map.find(tensor_type);
  if (it == type_map.end()) {
    throw std::runtime_error("No corresponding Numpy type for Tensor Type.");
  } else {
    return it->second;
  }
}

const MLDataType& NumpyToLotusTensorType(int numpy_type) {
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
      {NPY_UNICODE, DataTypeImpl::GetType<std::string>()}};

  if (numpy_type == NPY_STRING) {
    throw std::runtime_error("Please use np.unicode for strings.");
  }

  const auto it = type_map.find(numpy_type);
  if (it == type_map.end()) {
    throw std::runtime_error("Numpy_type " + std::to_string(numpy_type) +
                             " can't be converted to MLDataType.");
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
    const int npy_type = PyArray_TYPE(darray);

    // numpy requires long int as its dims.
    int ndim = PyArray_NDIM(darray);
    npy_intp* npy_dims = PyArray_DIMS(darray);
    std::vector<int64_t> dims;
    for (int i = 0; i < ndim; ++i) {
      dims.push_back(npy_dims[i]);
    }

    TensorShape shape(dims);
    auto element_type = NumpyToLotusTensorType(npy_type);
    void* buffer = alloc->Alloc(element_type->Size() * shape.Size());

    if (npy_type != NPY_UNICODE) {
      memcpy(buffer, static_cast<void*>(PyArray_DATA(darray)), element_type->Size() * shape.Size());
    }

    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                                shape,
                                                                static_cast<void*>(buffer),
                                                                alloc->Info(), alloc);

    if (npy_type == NPY_UNICODE) {
      // Copy string data which needs to be done after Tensor is allocated.
      std::string* dst = static_cast<std::string*>(buffer);
      auto item_size = PyArray_ITEMSIZE(darray);
      auto num_chars = item_size / PyUnicode_4BYTE_KIND;
      char* src = static_cast<char*>(PyArray_DATA(darray));
      for (int i = 0; i < shape.Size(); i++, src += item_size) {
        // Python unicode strings are assumed to be USC-4. Lotus strings are stored as UTF-8.
        dst[i] = PyUnicode_AsUTF8(PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, src, num_chars));
      }
    }

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

    throw;
  }

  if (!dref) {
    Py_XDECREF(darray);
  }
}

template <typename T>
void AddNonTensor(Lotus::MLValue& val, vector<py::object>& pyobjs) {
  pyobjs.push_back(py::cast(val.Get<T>()));
}
void AddNonTensorAsPyObj(Lotus::MLValue& val, vector<py::object>& pyobjs) {
  // Should be in sync with core/framework/datatypes.h
  if (val.Type() == DataTypeImpl::GetType<MapStringToString>()) {
    AddNonTensor<MapStringToString>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapStringToInt64>()) {
    AddNonTensor<MapStringToInt64>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapStringToFloat>()) {
    AddNonTensor<MapStringToFloat>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapStringToDouble>()) {
    AddNonTensor<MapStringToDouble>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapInt64ToString>()) {
    AddNonTensor<MapInt64ToString>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapInt64ToInt64>()) {
    AddNonTensor<MapInt64ToInt64>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapInt64ToFloat>()) {
    AddNonTensor<MapInt64ToFloat>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<MapInt64ToDouble>()) {
    AddNonTensor<MapInt64ToDouble>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorString>()) {
    AddNonTensor<VectorString>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorInt64>()) {
    AddNonTensor<VectorInt64>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorFloat>()) {
    AddNonTensor<VectorFloat>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorDouble>()) {
    AddNonTensor<VectorDouble>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorMapStringToFloat>()) {
    AddNonTensor<VectorMapStringToFloat>(val, pyobjs);
  } else if (val.Type() == DataTypeImpl::GetType<VectorMapInt64ToFloat>()) {
    AddNonTensor<VectorMapInt64ToFloat>(val, pyobjs);
  } else {
    throw std::runtime_error("Output is a non-tensor type which is not supported.");
  }
}

void AddTensorAsPyObj(Lotus::MLValue& val, vector<py::object>& pyobjs) {
  const Tensor& rtensor = val.Get<Tensor>();
  std::vector<npy_intp> npy_dims;
  const TensorShape& shape = rtensor.Shape();

  for (size_t n = 0; n < shape.NumDimensions(); ++n) {
    npy_dims.push_back(shape[n]);
  }

  MLDataType dtype = rtensor.DataType();
  const int numpy_type = LotusTensorToNumpyType(dtype);
  py::object obj = py::reinterpret_steal<py::object>(PyArray_SimpleNew(
      shape.NumDimensions(), npy_dims.data(), numpy_type));

  void* outPtr = static_cast<void*>(
      PyArray_DATA(reinterpret_cast<PyArrayObject*>(obj.ptr())));

  if (numpy_type != NPY_OBJECT) {
    memcpy(outPtr, rtensor.DataRaw(dtype), dtype->Size() * shape.Size());
  } else {
    // Handle string type.
    py::object* outObj = static_cast<py::object*>(outPtr);
    const std::string* src = rtensor.template Data<std::string>();
    for (int i = 0; i < rtensor.Shape().Size(); i++, src++) {
      outObj[i] = py::cast(*src);
    }
  }
  pyobjs.push_back(obj);
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
    static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CErrSink{}},
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
  // allow unit tests to redirect std::cout and std::cerr to sys.stdout and sys.stderr
  py::add_ostream_redirect(m, "lotus_ostream_redirect");
  py::class_<SessionOptions>(m, "SessionOptions")
      .def(py::init())
      .def_readwrite("enable_sequential_execution", &SessionOptions::enable_sequential_execution)
      .def_readwrite("enable_profiling", &SessionOptions::enable_profiling)
      .def_readwrite("profile_file_prefix", &SessionOptions::profile_file_prefix)
      .def_readwrite("session_logid", &SessionOptions::session_logid)
      .def_readwrite("session_log_verbosity_level", &SessionOptions::session_log_verbosity_level)
      .def_readwrite("enable_mem_pattern", &SessionOptions::enable_mem_pattern)
      .def_readwrite("max_num_graph_transformation_steps", &SessionOptions::max_num_graph_transformation_steps)
      .def_readwrite("enable_cpu_mem_arena", &SessionOptions::enable_cpu_mem_arena);

  py::class_<RunOptions>(m, "RunOptions")
      .def(py::init())
      .def_readwrite("run_log_verbosity_level", &RunOptions::run_log_verbosity_level)
      .def_readwrite("run_tag", &RunOptions::run_tag);

  py::class_<ModelMetadata>(m, "ModelMetadata")
      .def_readwrite("producer_name", &ModelMetadata::producer_name)
      .def_readwrite("graph_name", &ModelMetadata::graph_name)
      .def_readwrite("domain", &ModelMetadata::domain)
      .def_readwrite("description", &ModelMetadata::description)
      .def_readwrite("version", &ModelMetadata::version)
      .def_readwrite("custom_metadata_map", &ModelMetadata::custom_metadata_map);

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
      .def(py::init<SessionOptions, SessionObjectInitializer>())
      .def("load_model", [](InferenceSession* sess, const std::string& path) {
        auto status = sess->Load(path);

        if (!status.IsOK()) {
          throw std::runtime_error(status.ToString().c_str());
        }

        status = sess->Initialize();
        if (!status.IsOK()) {
          throw std::runtime_error(status.ToString().c_str());
        }
      })
      .def("run", [](InferenceSession* sess, std::vector<std::string> output_names, std::map<std::string, py::object> pyfeeds, RunOptions* run_options = nullptr) -> std::vector<py::object> {
        NameMLValMap feeds;
        for (auto _ : pyfeeds) {
          MLValue ml_value;
          PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(_.second.ptr());
          // Assumes Feeds are numpy arrays which we create tensors from.
          // TODO: support non-tensor type feeds.
          CreateTensorMLValue(GetAllocator(), arr, &ml_value);
          feeds.insert(std::make_pair(_.first, ml_value));
        }

        std::vector<MLValue> fetches;
        Common::Status status;

        if (run_options != nullptr) {
          status = sess->Run(*run_options, feeds, output_names, &fetches);
        } else {
          status = sess->Run(feeds, output_names, &fetches);
        }

        if (!status.IsOK()) {
          throw std::runtime_error(status.ToString().c_str());
        }

        std::vector<py::object> rfetch;
        rfetch.reserve(fetches.size());
        for (auto _ : fetches) {
          if (_.IsTensor()) {
            AddTensorAsPyObj(_, rfetch);
          } else {
            AddNonTensorAsPyObj(_, rfetch);
          }
        }
        return rfetch;
      })
      .def("end_profiling", [](InferenceSession* sess) -> std::string {
        return sess->EndProfiling();
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
      })
      .def_property_readonly("model_meta", [](const InferenceSession* sess) -> const Lotus::ModelMetadata& {
        auto res = sess->GetModelMetadata();
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

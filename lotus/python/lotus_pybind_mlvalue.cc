#include "lotus_pybind_mlvalue.h"

#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL lotus_python_ARRAY_API
#include <numpy/arrayobject.h>


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

bool PyObjectCheck_Array(PyObject* o) {
  return PyObject_HasAttrString(o, "__array_finalize__");
}

void CreateTensorMLValue(AllocatorPtr alloc, PyArrayObject* pyObject, MLValue* p_mlvalue) {
  PyArrayObject* darray = PyArray_GETCONTIGUOUS(pyObject);
  if (darray == NULL) {
    throw std::runtime_error("The object must be a contiguous array.");
  }
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

void CreateMapMLValue_LoopIntoMapInt64ToFloat(Py_ssize_t& pos, PyObject*& key, PyObject*& value,
                                              PyObject* item, MapInt64ToFloat& current) {
  long pylong;
  std::string cstr;
  do {
    pylong = PyLong_AsLong(key);

    if (PyFloat_Check(value)) {
      current[pylong] = (float)PyFloat_AS_DOUBLE(value);
    } else if (PyNumber_Check(value)) {
      current[pylong] = (float)PyFloat_AsDouble(value);
    } else {
      PyObject* pType = PyObject_Type(value);
      auto pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pStr);
      Py_XDECREF(pType);

      pStr = PyObject_Str(value);
      spyType = py::reinterpret_borrow<py::str>(pStr);
      sType += " - ";
      sType += spyType;
      Py_XDECREF(pStr);

      Py_XDECREF(item);
      throw std::runtime_error(std::string("Input is a list of dictionaries and they must have numbers as value (not ") + sType +
                               std::string(")"));
    }

  } while (PyDict_Next(item, &pos, &key, &value));
}

void CreateMapMLValue_LoopIntoMapStringToFloat(Py_ssize_t& pos, PyObject*& key, PyObject*& value,
                                               PyObject * item, MapStringToFloat &current) {

  PyObject* pStr;
  std::string cstr;
  do {
    pStr = PyObject_Str(key);
    cstr = py::reinterpret_borrow<py::str>(pStr);
    Py_XDECREF(pStr);

    if (PyFloat_Check(value)) {
      current[cstr] = (float)PyFloat_AS_DOUBLE(value);
    } else if (PyNumber_Check(value)) {
      current[cstr] = (float)PyFloat_AsDouble(value);
    } else {
      PyObject* pType = PyObject_Type(value);
      pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pStr);
      Py_XDECREF(pType);

      pStr = PyObject_Str(value);
      spyType = py::reinterpret_borrow<py::str>(pStr);
      sType += " - ";
      sType += spyType;
      Py_XDECREF(pStr);

      Py_XDECREF(item);
      throw std::runtime_error(std::string("Input is a list of dictionaries and they must have numbers as value (not ") + sType +
                               std::string(")"));
    }

  } while (PyDict_Next(item, &pos, &key, &value));
}

void CreateMapMLValue_MapToFloat(Py_ssize_t& pos, PyObject*& key, PyObject*& value,
                                 PyObject* iterator, PyObject* item,
                                 AllocatorPtr alloc, MLValue* p_mlvalue) {
  // If iterator is NULL, it returns a single MapStringToDouble,
  // if is not NULL, it returns a VectorMapStringToDouble.

  if (iterator == NULL) {
    if (PyLong_Check(key)){
      // TODO: write templating function with accessor.
      std::unique_ptr<MapInt64ToFloat> dst;
      dst = std::make_unique<MapInt64ToFloat>();
      CreateMapMLValue_LoopIntoMapInt64ToFloat(pos, key, value, item, *dst);
      p_mlvalue->Init(dst.release(), DataTypeImpl::GetType<MapInt64ToFloat>(),
                      DataTypeImpl::GetType<MapInt64ToFloat>()->GetDeleteFunc());
    } else if (PyUnicode_Check(key)) {
      std::unique_ptr<MapStringToFloat> dst;
      dst = std::make_unique<MapStringToFloat>();
      CreateMapMLValue_LoopIntoMapStringToFloat(pos, key, value, item, *dst);
      p_mlvalue->Init(dst.release(), DataTypeImpl::GetType<MapStringToFloat>(),
                      DataTypeImpl::GetType<MapStringToFloat>()->GetDeleteFunc());
    } else {
      PyObject* pType = PyObject_Type(key);
      PyObject* pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pType);
      Py_XDECREF(pStr);
      throw std::runtime_error(std::string("Key type must be int or string (not ") + sType +
                               std::string(" )."));
    }
  } else {
    if (PyLong_Check(value)) {
      std::unique_ptr<VectorMapInt64ToFloat> dstVector;
      dstVector = std::make_unique<VectorMapInt64ToFloat>();
      int index = 0;
      do {
        dstVector->push_back(MapInt64ToFloat());
        CreateMapMLValue_LoopIntoMapInt64ToFloat(pos, key, value, item, (*dstVector)[index]);
        Py_DECREF(item);
        ++index;
        item = iterator == NULL ? NULL : PyIter_Next(iterator);
      } while (item != NULL);
      p_mlvalue->Init(dstVector.release(), DataTypeImpl::GetType<VectorMapInt64ToFloat>(),
                      DataTypeImpl::GetType<VectorMapInt64ToFloat>()->GetDeleteFunc());
    } else if (PyUnicode_Check(value)) {
      std::unique_ptr<VectorMapStringToFloat> dstVector;
      dstVector = std::make_unique<VectorMapStringToFloat>();
      int index = 0;
      do {
        dstVector->push_back(MapStringToFloat());
        CreateMapMLValue_LoopIntoMapStringToFloat(pos, key, value, item, (*dstVector)[index]);
        Py_DECREF(item);
        ++index;
        item = iterator == NULL ? NULL : PyIter_Next(iterator);
      } while (item != NULL);
      p_mlvalue->Init(dstVector.release(), DataTypeImpl::GetType<VectorMapStringToFloat>(),
                      DataTypeImpl::GetType<VectorMapStringToFloat>()->GetDeleteFunc());
    } else {
      PyObject* pType = PyObject_Type(value);
      PyObject* pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pType);
      Py_XDECREF(pStr);
      throw std::runtime_error(std::string("Value type must be int or string (not ") + sType +
                               std::string(" )."));
    }
  }
}

void CreateMapMLValue_VectorMapToFloat(PyObject* iterator, PyObject* item, AllocatorPtr alloc, MLValue* p_mlvalue) {
  // CreateMapMLValue is called by CreateGenericTerableMLValue which ensures
  // item is a dictionary, no need to check type again.
  // Onnxmltools only uses dictionaries with vector as key type.
  // Lotus converts that into a map<string[1], ...>.
  // We assumes that two conditions hold or the following code
  // raises an exception. That also relies on the fact the conversion
  // from an array of int into string happens the same in onnxmltools
  // and the following code (meaning option int64_vocabulary is never used).

  // If iterator is NULL, it returns a single MapStringToDouble,
  // if is not NULL, it returns a VectorMapStringToDouble.

  PyObject *key, *value;
  Py_ssize_t pos = 0;

  if (PyDict_Next(item, &pos, &key, &value)) {
    if (PyFloat_Check(value)) {
      CreateMapMLValue_MapToFloat(pos, key, value, iterator, item, alloc, p_mlvalue);
    } else {
      PyObject* pType = PyObject_Type(value);
      PyObject* pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pType);
      Py_XDECREF(pStr);
      throw std::runtime_error(std::string("Unable convert object of type ") + sType +
                               std::string(" into a tensor."));
    }
  } else {
    throw std::runtime_error("Size of dictionary is empty, unable to run the prediction.");
  }
}

void CreateGenericIterableMLValue(PyObject* iterator, AllocatorPtr alloc, MLValue* p_mlvalue) {
  PyObject* item;
  MLValue ml_value;
  item = PyIter_Next(iterator);
  if (item == NULL) {
    throw std::runtime_error("Inputs must not be empty.");
  }
  if (PyObjectCheck_Array(item)) {
    PyObject* pType = PyObject_Type(item);
    PyObject* pStr = PyObject_Str(pType);
    py::str spyType = py::reinterpret_borrow<py::str>(pStr);
    std::string sType = spyType;
    Py_XDECREF(pType);
    Py_XDECREF(pStr);
    throw std::runtime_error("Iterable of " + sType + " should be given as array");
  } else {
    // We expect a dictionary.
    if (!PyDict_Check(item)) {
      throw std::runtime_error("Input must be a list of dictionaries or a single numpy array.");
    }
    CreateMapMLValue_VectorMapToFloat(iterator, item, alloc, p_mlvalue);
  }
}

void CreateGenericMLValue(AllocatorPtr alloc, py::object& value, MLValue* p_mlvalue) {
  if (PyObjectCheck_Array(value.ptr())) {
    // The most frequent case: input comes as an array.
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(value.ptr());
    CreateTensorMLValue(alloc, arr, p_mlvalue);
  } else if (PyDict_Check(value.ptr())) {
    // One single input of a DictVectorizer.
    CreateMapMLValue_VectorMapToFloat((PyObject*)NULL, value.ptr(), alloc, p_mlvalue);
  } else {
    // An enumerator of inputs for a DictVectorizer.
    auto iterator = PyObject_GetIter(value.ptr());
    if (iterator == NULL) {
      // The pype cannot be handled.
      PyObject* pType = PyObject_Type(value.ptr());
      PyObject* pStr = PyObject_Str(pType);
      py::str spyType = py::reinterpret_borrow<py::str>(pStr);
      std::string sType = spyType;
      Py_XDECREF(pType);
      Py_XDECREF(pStr);
      throw std::runtime_error(std::string("Unable to handle object of type ") + sType);
    }
    // We assume the object is iterable.
    // iterator should not be NULL due to previous test.
    try {
      CreateGenericIterableMLValue(iterator, alloc, p_mlvalue);
    } catch (std::runtime_error e) {
      Py_DECREF(iterator);
      throw;
    }
    Py_DECREF(iterator);
  }
}

}  // namespace python
}  // namespace lotus

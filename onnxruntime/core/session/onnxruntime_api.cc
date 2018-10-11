// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"  //TODO: remove this
#include "core/session/onnxruntime_c_api.h"
#include "core/session/allocator_impl.h"
#include <cassert>
#include <cstring>
#include <sstream>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/status.h"
#include "core/inc/op_kernel_author.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/ml_value.h"
#include "core/framework/environment.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/inference_session.h"
#include "core/graph/graph_base.h"
#ifdef USE_CUDA
#include "core/providers/provider_factories.h"
#endif
#include "abi_session_options_impl.h"

using namespace onnxruntime::logging;
using onnxruntime::DataTypeImpl;
using onnxruntime::Environment;
using onnxruntime::IAllocator;
using onnxruntime::InputDefList;
using onnxruntime::MLFloat16;
using onnxruntime::MLStatus;
using onnxruntime::MLValue;
using onnxruntime::OutputDefList;
using onnxruntime::Tensor;
using onnxruntime::common::Status;

#define ONNXRUNTIME_API_RETURN_IF_ERROR(expr) \
  do {                                        \
    auto _status = (expr);                    \
    if (_status) return _status;              \
  } while (0)

struct ONNXEnv {
  ONNXEnv(Environment* value1, LoggingManager* loggingManager1) : value(value1), loggingManager(loggingManager1) {
  }
  /**
  * This function will call ::google::protobuf::ShutdownProtobufLibrary
  */
  ~ONNXEnv() {
    delete loggingManager;
    delete value;
  }
  Environment* value;
  LoggingManager* loggingManager;
  ONNXRUNTIME_DISALLOW_COPY_AND_ASSIGNMENT(ONNXEnv);
};

static ONNXStatusPtr ToONNXStatus(const Status& st) {
  if (st.IsOK())
    return nullptr;
  size_t clen = st.ErrorMessage().length();
  size_t len = clen + 1 + sizeof(int);
  char* p = new char[len];
  char* ret = p;
  *reinterpret_cast<int*>(p) = static_cast<int>(st.Code());
  p += sizeof(int);
  memcpy(p, st.ErrorMessage().c_str(), clen);
  p += clen;
  *p = '\0';
  return ret;
}

ONNXRUNTIME_API(ONNXRuntimeErrorCode, ONNXRuntimeGetErrorCode, _In_ const ONNXStatusPtr status) {
  return *reinterpret_cast<ONNXRuntimeErrorCode*>(status);
}

ONNXRUNTIME_API(const char*, ONNXRuntimeGetErrorMessage, _In_ const ONNXStatusPtr status) {
  return reinterpret_cast<const char*>(status) + sizeof(int);
}

static ONNXStatusPtr CreateONNXStatus(ONNXRuntimeErrorCode code, const char* msg) {
#ifndef NDEBUG
  assert(!(code == 0 && msg != nullptr));
#endif
  size_t clen = strlen(msg);
  size_t len = clen + 1 + sizeof(int);
  char* p = new char[len];
  char* ret = p;
  *reinterpret_cast<int*>(p) = static_cast<int>(code);
  p += sizeof(int);
  memcpy(p, msg, clen);
  p += clen;
  *p = '\0';
  return ret;
}

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                                   \
  }                                                                    \
  catch (std::exception & ex) {                                        \
    return CreateONNXStatus(ONNXRUNTIME_RUNTIME_EXCEPTION, ex.what()); \
  }

#define TENSOR_READ_API_BEGIN                                \
  API_IMPL_BEGIN                                             \
  auto v = reinterpret_cast<::onnxruntime::MLValue*>(value); \
  auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN                           \
  API_IMPL_BEGIN                                             \
  auto v = reinterpret_cast<::onnxruntime::MLValue*>(value); \
  auto tensor = v->GetMutable<onnxruntime::Tensor>();

class LoggingWrapper : public ISink {
 public:
  LoggingWrapper(ONNXRuntimeLoggingFunction logging_function, void* logger_param)
      : logging_function_{logging_function}, logger_param_{logger_param} {
  }

  void SendImpl(const Timestamp& /*timestamp*/ /*timestamp*/, const std::string& logger_id, const Capture& message) override {
    std::string s = message.Location().ToString();
    logging_function_(logger_param_, static_cast<ONNXRuntimeLoggingLevel>(message.Severity()), message.Category(), logger_id.c_str(), s.c_str(), message.Message().c_str());
  }

 private:
  ONNXRuntimeLoggingFunction logging_function_;
  void* logger_param_;
};

ONNXRUNTIME_API_STATUS_IMPL(InitializeONNXRuntimeWithCustomLogger, ONNXRuntimeLoggingFunction logging_function, void* logger_param, ONNXRuntimeLoggingLevel default_warning_level, _In_ const char* logid, _Out_ ONNXEnv** out) {
  API_IMPL_BEGIN
  std::string name = logid;
  std::unique_ptr<ISink> logger = std::make_unique<LoggingWrapper>(logging_function, logger_param);
  std::unique_ptr<LoggingManager> default_logging_manager = std::make_unique<LoggingManager>(std::move(logger),
                                                                                             static_cast<Severity>(default_warning_level), false,
                                                                                             LoggingManager::InstanceType::Default,
                                                                                             &name);
  std::unique_ptr<Environment> env;
  Status status = Environment::Create(env);
  if (status.IsOK())
    *out = new ONNXEnv(env.release(), default_logging_manager.release());
  return ToONNXStatus(status);
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(InitializeONNXRuntime, ONNXRuntimeLoggingLevel default_warning_level, _In_ const char* logid, _Out_ ONNXEnv** out) {
  API_IMPL_BEGIN
  std::string name = logid;
  std::unique_ptr<LoggingManager> default_logging_manager = std::make_unique<LoggingManager>(std::unique_ptr<ISink>{new CLogSink{}},
                                                                                             static_cast<Severity>(default_warning_level), false,
                                                                                             LoggingManager::InstanceType::Default,
                                                                                             &name);
  std::unique_ptr<Environment> env;
  Status status = Environment::Create(env);
  if (status.IsOK())
    *out = new ONNXEnv(env.release(), default_logging_manager.release());
  return ToONNXStatus(status);
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetStringTensorDataLength, _In_ ONNXValuePtr value, _Out_ size_t* out) {
  TENSOR_READ_API_BEGIN
  const auto* src = tensor.Data<std::string>();
  int64_t len = tensor.Shape().Size();
  if (len >= 0) {
    size_t ret = 0;
    for (int64_t i = 0; i != len; ++i) {
      ret += src[i].size();
    }
    *out = ret;
  } else
    return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, "shape is invalid");
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeFillStringTensor, _In_ ONNXValuePtr value, _In_ const char* s[], size_t s_len) {
  TENSOR_READWRITE_API_BEGIN
  auto* dst = tensor->MutableData<std::string>();
  auto len = static_cast<size_t>(tensor->Shape().Size());
  if (s_len < len) {
    return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, "input array is too short");
  }
  for (size_t i = 0; i != len; ++i) {
    //allocate and copy
    dst[i] = s[i];
  }
  return nullptr;
  API_IMPL_END
}

template <typename T>
void CreateTensorImpl(const size_t* shape, size_t shape_len, std::shared_ptr<onnxruntime::IAllocator>& allocator, std::unique_ptr<Tensor>* out) {
  size_t elem_count = 1;
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    elem_count *= shape[i];
    shapes[i] = shape[i];
  }

  size_t size_to_allocate = sizeof(T) * elem_count;
  void* p_data = allocator->Alloc(size_to_allocate);
  *out = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                  onnxruntime::TensorShape(shapes.data(), shape_len),
                                  static_cast<void*>(p_data),
                                  allocator->Info(),
                                  allocator);
}

template <typename T>
ONNXStatusPtr CreateTensorImpl(const size_t* shape, size_t shape_len, const ONNXRuntimeAllocatorInfo* info, void* p_data, size_t p_data_len, std::unique_ptr<Tensor>* out) {
  size_t elem_count = 1;
  std::vector<int64_t> shapes(shape_len);
  for (size_t i = 0; i != shape_len; ++i) {
    elem_count *= shape[i];
    shapes[i] = shape[i];
  }

  size_t size_to_allocate = sizeof(T) * elem_count;
  if (size_to_allocate > p_data_len) {
    std::ostringstream oss;
    oss << "not enough space: expected " << size_to_allocate << ", got " << p_data_len;
    return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, oss.str().c_str());
  }
  *out = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                  onnxruntime::TensorShape(shapes.data(), shape_len),
                                  p_data,
                                  *(onnxruntime::AllocatorInfo*)info,
                                  nullptr);
  return nullptr;
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateTensorWithDataAsONNXValue, _In_ const ONNXRuntimeAllocatorInfo* info, _In_ void* p_data, size_t p_data_len, _In_ const size_t* shape, size_t shape_len, OnnxRuntimeTensorElementDataType type, _Out_ ONNXValuePtr* out) {
  API_IMPL_BEGIN
  std::unique_ptr<Tensor> tensor;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<float>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<uint8_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<int8_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<uint16_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<int16_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<int32_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<int64_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<std::string>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<bool>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<MLFloat16>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<double>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<uint32_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      ONNXRUNTIME_API_RETURN_IF_ERROR(CreateTensorImpl<uint64_t>(shape, shape_len, info, p_data, p_data_len, &tensor));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    default: {
      std::ostringstream oss;
      oss << "type " << type << " is not supported in this function";
      std::string errmsg = oss.str();
      return CreateONNXStatus(ONNXRUNTIME_NOT_IMPLEMENTED, errmsg.c_str());
    }
  }
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(tensor.release(),
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  *out = reinterpret_cast<ONNXValuePtr>(value.release());
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateTensorAsONNXValue, _Inout_ ONNXRuntimeAllocator* allocator, _In_ const size_t* shape, size_t shape_len, OnnxRuntimeTensorElementDataType type, _Out_ ONNXValuePtr* out) {
  API_IMPL_BEGIN
  std::shared_ptr<onnxruntime::IAllocator> allocator_ = std::make_shared<onnxruntime::AllocatorWrapper>(allocator);
  std::unique_ptr<Tensor> tensor;
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      CreateTensorImpl<float>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      CreateTensorImpl<uint8_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      CreateTensorImpl<int8_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      CreateTensorImpl<uint16_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      CreateTensorImpl<int16_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      CreateTensorImpl<int32_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      CreateTensorImpl<int64_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      CreateTensorImpl<std::string>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      CreateTensorImpl<bool>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      CreateTensorImpl<MLFloat16>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      CreateTensorImpl<double>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      CreateTensorImpl<uint32_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      CreateTensorImpl<uint64_t>(shape, shape_len, allocator_, &tensor);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    default: {
      std::ostringstream oss;
      oss << "type " << type << " is not supported in this function";
      std::string errmsg = oss.str();
      return CreateONNXStatus(ONNXRUNTIME_NOT_IMPLEMENTED, errmsg.c_str());
    }
  }
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  value->Init(tensor.release(),
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  *out = reinterpret_cast<ONNXValuePtr>(value.release());
  return nullptr;
  API_IMPL_END
}

template <typename T>
static ONNXStatusPtr CreateInferenceSessionImpl(_In_ ONNXEnv* env, _In_ T model_path, _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSessionPtr* out) {
  API_IMPL_BEGIN
  std::unique_ptr<::onnxruntime::InferenceSession> sess = std::make_unique<::onnxruntime::InferenceSession>(options->value, env->loggingManager);
  Status status;
  if (!options->custom_op_paths.empty()) {
    status = sess->LoadCustomOps(options->custom_op_paths);
    if (!status.IsOK())
      return ToONNXStatus(status);
  }
  if (options->enable_cuda_provider) {
    //TODO: instead of creating providers at there, it should get a functor from ONNXEnv and call it
#if USE_CUDA
    onnxruntime::CUDAExecutionProviderInfo cuda_pi;
    cuda_pi.device_id = options->cuda_device_id;
    status = sess->RegisterExecutionProvider(onnxruntime::CreateCUDAExecutionProvider(cuda_pi));
    if (!status.IsOK())
      return ToONNXStatus(status);
#else
    return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, "This executable was not built with CUDA");
#endif
  }
  status = sess->Load(model_path);
  if (!status.IsOK())
    return ToONNXStatus(status);
  status = sess->Initialize();
  if (!status.IsOK())
    return ToONNXStatus(status);
  *out = reinterpret_cast<ONNXSessionPtr>(sess.release());
  return nullptr;
  API_IMPL_END
}

#ifdef _WIN32
ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateInferenceSession, _In_ ONNXEnv* env, _In_ const wchar_t* model_path, _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSessionPtr* out) {
  API_IMPL_BEGIN
  return CreateInferenceSessionImpl(env, model_path, options, out);
  API_IMPL_END
}
#else
ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateInferenceSession, _In_ ONNXEnv* env, _In_ const char* model_path, _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSessionPtr* out) {
  API_IMPL_BEGIN
  return CreateInferenceSessionImpl(env, model_path, options, out);
  API_IMPL_END
}
#endif

ONNXRUNTIME_API_STATUS_IMPL(RunInferenceAndFetchAll, _In_ ONNXSessionPtr sess, _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len, _Out_ ONNXValueListPtr* output, _Out_ size_t* output_len) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  ::onnxruntime::NameMLValMap in;
  for (size_t i = 0; i != input_len; ++i) {
    auto kvp = in.insert(std::make_pair(std::string(input_names[i]), *reinterpret_cast<::onnxruntime::MLValue*>(input[i])));
    if (!kvp.second) {
      return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, "duplicated input name");
    }
  }
  // Create output feed
  std::vector<std::string> output_names;
  for (auto const& outp : *(session->GetModelOutputs().second)) {
    output_names.push_back(outp->Name());
  }
  std::vector<MLValue> fetches;
  auto status = session->Run(in, output_names, &fetches);
  if (!status.IsOK())
    return ToONNXStatus(status);
  auto* out = new MLValue[fetches.size()];
  const int queue_id = 0;
  for (size_t i = 0; i != fetches.size(); ++i) {
    if (fetches[i].Fence())
      fetches[i].Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
    out[i] = fetches[i];
  }
  *output_len = fetches.size();
  *output = reinterpret_cast<ONNXValueListPtr>(out);
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetTensorMutableData, _In_ ONNXValuePtr value, _Out_ void** output) {
  TENSOR_READWRITE_API_BEGIN
  //TODO: test if it's a string tensor
  *output = tensor->MutableDataRaw();
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetTensorShape, _In_ ONNXValuePtr value, _Out_ size_t* shape_array, size_t shape_array_len) {
  TENSOR_READ_API_BEGIN
  const onnxruntime::TensorShape& shape = tensor.Shape();
  size_t len = shape.NumDimensions();
  if (shape_array_len < len) {
    std::ostringstream oss;
    oss << "input array doesn't have enough rooom, needs " << len;
    std::string errmsg = oss.str();
    return CreateONNXStatus(ONNXRUNTIME_INVALID_ARGUMENT, errmsg.c_str());
  }
  memcpy(shape_array, shape.GetDims().data(), len * sizeof(size_t));
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetTensorShapeDimCount, ONNXValuePtr value, size_t* out) {
  TENSOR_READ_API_BEGIN
  const onnxruntime::TensorShape& shape = tensor.Shape();
  *out = shape.NumDimensions();
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetTensorShapeElementCount, _In_ ONNXValuePtr value, _Out_ size_t* out) {
  TENSOR_READ_API_BEGIN
  const onnxruntime::TensorShape& shape = tensor.Shape();
  *out = shape.Size();
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeGetStringTensorContent, _In_ ONNXValuePtr value, _Out_ void* s, size_t s_len, _Out_ size_t* offsets, size_t offsets_len) {
  TENSOR_READ_API_BEGIN
  const auto* input = tensor.Data<std::string>();
  auto len = static_cast<size_t>(tensor.Shape().Size());
  if (offsets_len < len) {
    return CreateONNXStatus(ONNXRUNTIME_FAIL, "space is not enough");
  }
  {
    size_t ret = 0;
    for (size_t i = 0; i != len; ++i) {
      ret += input[i].size();
    }
    if (s_len < ret) {
      return CreateONNXStatus(ONNXRUNTIME_FAIL, "space is not enough");
    }
  }
  size_t f = 0;
  char* p = static_cast<char*>(s);
  for (size_t i = 0; i != offsets_len; ++i, ++offsets) {
    memcpy(p, input[i].data(), input[i].size());
    p += input[i].size();
    *offsets = f;
    f += input[i].size();
  }
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeTensorProtoToONNXValue, _Inout_ ONNXRuntimeAllocator* allocator, const void* input, int input_len, _Out_ ONNXValuePtr* out) {
  API_IMPL_BEGIN
  std::shared_ptr<onnxruntime::IAllocator> allocator_ = std::make_shared<onnxruntime::AllocatorWrapper>(allocator);
  ::ONNX_NAMESPACE::TensorProto proto;
  if (!proto.ParseFromArray(input, input_len)) {
    return CreateONNXStatus(ONNXRUNTIME_FAIL, "parse input tensor proto failed");
  }
  std::unique_ptr<MLValue> value = std::make_unique<MLValue>();
  Status st = onnxruntime::utils::TensorProtoToMLValue(proto, allocator_, nullptr, 0, *value);
  if (!st.IsOK())
    return ToONNXStatus(st);
  *out = reinterpret_cast<ONNXValuePtr>(value.release());
  return nullptr;
  API_IMPL_END
}

ONNXRUNTIME_API(ONNXValuePtr, ONNXValueListGetNthValue, ONNXValueListPtr list, size_t index) {
  auto v = reinterpret_cast<::onnxruntime::MLValue*>(list);
  return reinterpret_cast<ONNXValuePtr>(v + index);
}

#define DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION(INPUT_TYPE, REAL_TYPE) \
  ONNXRUNTIME_API(void, Release##INPUT_TYPE, INPUT_TYPE##Ptr value) {      \
    delete reinterpret_cast<REAL_TYPE*>(value);                            \
  }

#define DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION_FOR_ARRAY(INPUT_TYPE, REAL_TYPE) \
  ONNXRUNTIME_API(void, Release##INPUT_TYPE, INPUT_TYPE##Ptr value) {                \
    delete[] reinterpret_cast<REAL_TYPE*>(value);                                    \
  }

ONNXRUNTIME_API(void, ReleaseONNXEnv, ONNXEnv* env) {
  delete env;
}

DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION(ONNXValue, MLValue)
DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION(ONNXSession, ::onnxruntime::InferenceSession)
DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION_FOR_ARRAY(ONNXValueList, ::onnxruntime::MLValue)
DEFINE_RELEASE_ONNX_RUNTIME_OBJECT_FUNCTION_FOR_ARRAY(ONNXStatus, char)

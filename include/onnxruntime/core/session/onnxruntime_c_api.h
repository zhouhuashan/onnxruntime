// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

//TODO:provider order

#include "core/common/visibility_macros.h"
#include "error_code.h"
#include "allocator.h"

#ifdef __cplusplus
extern "C" {
#endif

//Any pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.

//copied from TensorProto::DataType
typedef enum OnnxRuntimeTensorElementDataType {
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1,   // float
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2,   // uint8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3,    // int8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4,  // uint16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5,   // int16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6,   // int32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7,   // int64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8,  // string
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,    // bool
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14,   // complex with float32 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15,  // complex with float64 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16,    // Non-IEEE floating-point format based on IEEE754 single-precision
  ONNX_TENSOR_ELEMENT_DATA_TYPE_MAX = 17
} OnnxRuntimeTensorElementDataType;

typedef enum ONNXRuntimeType {
  ONNXRUNTIME_TYPE_TENSOR,
  ONNXRUNTIME_TYPE_SEQUENCE,
  ONNXRUNTIME_TYPE_MAP,
  ONNXRUNTIME_TYPE_OPAQUE,
  ONNXRUNTIME_TYPE_ELEMENT,  //basic types like float/int32
} ONNXRuntimeType;

typedef struct ONNXArray {
  OnnxRuntimeTensorElementDataType type;
  size_t length;
  void* data;
} ONNXArray;

typedef struct ONNXTensorTypeInfo {
  OnnxRuntimeTensorElementDataType type;
  int64_t* dim_values;
  //length of dim_values
  size_t dim_count;
} ONNXTensorTypeInfo;

typedef struct ONNXOpaqueTypeInfo {
  char* domain;
  char* name;
} ONNXOpaqueTypeInfo;

//Each ONNX value is a n-ary tree.
//Data is only stored in leaf nodes.
//Every non-leaf node contains a field of ONNXRuntimeType
//Each leaf node is either a tensor, or an ONNXArray.

/**
 * ReleaseONNXEnv function calls ::google::protobuf::ShutdownProtobufLibrary().
 * Therefore, you should only call ReleaseONNXEnv at the end of your program.
 * Once you did that, don't call any onnxruntime, onnx or protobuf functions again.
 */
DEFINE_RUNTIME_CLASS(ONNXEnv);

DEFINE_RUNTIME_CLASS(ONNXRuntimeProvider);

typedef struct ONNXRuntimeProviderFactoryInterface {
  //These methods returns the new reference count.
  uint32_t (ONNXRUNTIME_API_STATUSCALL *AddRef)(void* this_);
  uint32_t (ONNXRUNTIME_API_STATUSCALL *Release)(void* this_);
  ONNXStatusPtr (ONNXRUNTIME_API_STATUSCALL *CreateProvider)(void* this_, ONNXRuntimeProviderPtr* out);
} ONNXRuntimeProviderFactoryInterface;

typedef ONNXRuntimeProviderFactoryInterface* ONNXRuntimeProviderFactoryPtr;

typedef enum ONNXRuntimeLoggingLevel {
  ONNXRUNTIME_LOGGING_LEVEL_kVERBOSE = 0,
  ONNXRUNTIME_LOGGING_LEVEL_kINFO = 1,
  ONNXRUNTIME_LOGGING_LEVEL_kWARNING = 2,
  ONNXRUNTIME_LOGGING_LEVEL_kERROR = 3,
  ONNXRUNTIME_LOGGING_LEVEL_kFATAL = 4
} ONNXRuntimeLoggingLevel;

typedef void (ONNXRUNTIME_API_STATUSCALL *ONNXRuntimeLoggingFunction)(void* param, ONNXRuntimeLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message);
/**
 * ONNXEnv is process-wise. For each process, only one ONNXEnv can be created. Don't do it multiple times
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeInitialize, ONNXRuntimeLoggingLevel default_warning_level, _In_ const char* logid, _Out_ ONNXEnv** out);
/**
 * ONNXEnv is process-wise. For each process, only one ONNXEnv can be created. Don't do it multiple times
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeInitializeWithCustomLogger, ONNXRuntimeLoggingFunction logging_function, void* logger_param, ONNXRuntimeLoggingLevel default_warning_level, _In_ const char* logid, _Out_ ONNXEnv** out);

DEFINE_RUNTIME_CLASS(ONNXRuntimeSessionOptions);

ONNXRUNTIME_API(ONNXRuntimeSessionOptions*, ONNXRuntimeCreateSessionOptions, void);

/// create a copy of an existing ONNXRuntimeSessionOptions
ONNXRUNTIME_API(ONNXRuntimeSessionOptions*, ONNXRuntimeCloneSessionOptions, ONNXRuntimeSessionOptions*);
ONNXRUNTIME_API(void, ONNXRuntimeEnableSequentialExecution, _In_ ONNXRuntimeSessionOptions* options);
ONNXRUNTIME_API(void, ONNXRuntimeDisableSequentialExecution, _In_ ONNXRuntimeSessionOptions* options);

// enable profiling for this session.
ONNXRUNTIME_API(void, ONNXRuntimeEnableProfiling, _In_ ONNXRuntimeSessionOptions* options, _In_ const char* profile_file_prefix);
ONNXRUNTIME_API(void, ONNXRuntimeDisableProfiling, _In_ ONNXRuntimeSessionOptions* options);

// enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
ONNXRUNTIME_API(void, ONNXRuntimeEnableMemPattern, _In_ ONNXRuntimeSessionOptions* options);
ONNXRUNTIME_API(void, ONNXRuntimeDisableMemPattern, _In_ ONNXRuntimeSessionOptions* options);

// enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ONNXRUNTIME_API(void, ONNXRuntimeEnableCpuMemArena, _In_ ONNXRuntimeSessionOptions* options);
ONNXRUNTIME_API(void, ONNXRuntimeDisableCpuMemArena, _In_ ONNXRuntimeSessionOptions* options);

///< logger id to use for session output
ONNXRUNTIME_API(void, ONNXRuntimeSetSessionLogId, _In_ ONNXRuntimeSessionOptions* options, const char* logid);

///< applies to session load, initialization, etc
ONNXRUNTIME_API(void, ONNXRuntimeSetSessionLogVerbosityLevel, _In_ ONNXRuntimeSessionOptions* options, uint32_t session_log_verbosity_level);

///How many threads in the session thread pool.
ONNXRUNTIME_API(int, ONNXRuntimeSetSessionThreadPoolSize, _In_ ONNXRuntimeSessionOptions* options, int session_thread_pool_size);

/**
  * The order of invocation indicates the preference order as well. In other words call this method
  * on your most preferred execution provider first followed by the less preferred ones.
  * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
  */
ONNXRUNTIME_API(void, ONNXRuntimeSessionOptionsAppendExecutionProvider, _In_ ONNXRuntimeSessionOptions* options, _In_ ONNXRuntimeProviderFactoryPtr* f);

ONNXRUNTIME_API(void, ONNXRuntimeAddCustomOp, _In_ ONNXRuntimeSessionOptions* options, const char* custom_op_path);

DEFINE_RUNTIME_CLASS(ONNXSession);

//TODO: document the path separator convention? '/' vs '\'
//TODO: should specify the access characteristics of model_path. Is this read only during the
//execution of ONNXRuntimeCreateInferenceSession, or does the ONNXSession retain a handle to the file/directory and continue to access throughout the ONNXSession lifetime?
// What sort of access is needed to model_path : read or read/write?
//TODO:  allow loading from an in-memory byte-array
#ifdef _WIN32
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateInferenceSession, _In_ ONNXEnv* env, _In_ const wchar_t* model_path, _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSessionPtr* out);
#else
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateInferenceSession, _In_ ONNXEnv* env, _In_ const char* model_path, _In_ const ONNXRuntimeSessionOptions* options, _Out_ ONNXSessionPtr* out);
#endif

DEFINE_RUNTIME_CLASS(ONNXValue);

/**
 * This function is only for advanced users. In most cases, please use ONNXRuntimeCreateTensorWithDataAsONNXValue
 * The returned ONNXValuePtr will keep a reference to allocator, without reference counting
 * \param type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateTensorAsONNXValue, _Inout_ ONNXRuntimeAllocator* allocator, _In_ const size_t* shape, size_t shape_len, OnnxRuntimeTensorElementDataType type, _Out_ ONNXValuePtr* out);

/**
 * p_data is owned by caller. ReleaseTensor won't release p_data. 
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateTensorWithDataAsONNXValue, _In_ const ONNXRuntimeAllocatorInfo* info, _In_ void* p_data, size_t p_data_len, _In_ const size_t* shape, size_t shape_len, OnnxRuntimeTensorElementDataType type, _Out_ ONNXValuePtr* out);

/// This function doesn't work with string tensor
/// this is a no-copy method whose pointer is only valid until the backing ONNXValuePtr is free'd.
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetTensorMutableData, _In_ ONNXValuePtr value, _Out_ void** out);

/**
 * \param value A tensor created from ONNXRuntimeCreateTensor*** function.
 * \param s each A string array. Each string in this array must be null terminated.
 * \param s_len length of s
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeFillStringTensor, _In_ ONNXValuePtr value, _In_ const char* s[], size_t s_len);
/**
 * \param value A tensor created from ONNXRuntimeCreateTensor*** function.
 * \param len total data length, not including the trailing '\0' chars.
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetStringTensorDataLength, _In_ ONNXValuePtr value, _Out_ size_t* len);

/**
 * \param s string contents. Each string is NOT null-terminated.
 * \param value A tensor created from ONNXRuntimeCreateTensor*** function.
 * \param s_len total data length, get it from ONNXRuntimeGetStringTensorDataLength
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetStringTensorContent, _In_ ONNXValuePtr value, _Out_ void* s, size_t s_len, _Out_ size_t* offsets, size_t offsets_len);

/**
 * If the tensor has shape of {3,4,5}, *out would be 3
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetTensorShapeDimCount, _In_ ONNXValuePtr, _Out_ size_t* out);

/**
 * Generally, user should call ONNXRuntimeGetTensorShapeDimCount before calling this.
 * Unless they already have a good estimation on the dimension count
 * \param shape_array An array allocated by caller, with size of shape_array
 * \param shape_array_len the length of passed in shape_array.
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetTensorShape, _In_ ONNXValuePtr, _Out_ size_t* shape_array, size_t shape_array_len);

/**
 * How many elements does this tensor have.
 * [1,3,4] -> 12
 * [2,0,4] -> 0
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeGetTensorShapeElementCount, _In_ ONNXValuePtr, _Out_ size_t* out);
//not implemented
//ONNX_RUNTIME_EXPORT int GetPONNXValueDataType(_In_ ONNXValuePtr) NO_EXCEPTION;

DEFINE_RUNTIME_CLASS(ONNXValueList);

//For InferenceSession run calls, all the input values shouldn't created by allocator
//User should manage the buffer by himself, not allocator

/**
 * \param sess created by ONNXRuntimeCreateInferenceSession function
 * \param output must be freed by ReleaseONNXValueListPtr function
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeRunInferenceAndFetchAll, _In_ ONNXSessionPtr sess, _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len, _Out_ ONNXValueListPtr* output, _Out_ size_t* output_len);
ONNXRUNTIME_API_STATUS(ONNXRuntimeRunInference, _In_ ONNXSessionPtr sess, _In_ const char* input_names[], _In_ ONNXValuePtr* input, size_t input_len, _In_ const char* output_names[], size_t output_names_len, _Out_ ONNXValueListPtr* output);

ONNXRUNTIME_API(int, ONNXRuntimeInferenceSessionGetInputCount, _In_ ONNXSessionPtr sess);
ONNXRUNTIME_API(int, ONNXRuntimeInferenceSessionGetOutputCount, _In_ ONNXSessionPtr sess);

//Tree for PONNXType:
//ONNXRUNTIME_TYPE_TENSOR -> ONNXTensorTypeInfo
//ONNXRUNTIME_TYPE_SEQUENCE -> nullptr
//ONNXRUNTIME_TYPE_MAP -> nullptr
//ONNXRUNTIME_TYPE_OPAQUE-> ONNXOpaqueTypeInfo
//ONNXRUNTIME_TYPE_ELEMENT -> nullptr

//The output value must be freed by ONNXRuntimeNodeDestoryTree
//ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetInputType, _In_ ONNXSessionPtr sess, _Out_ PONNXType* out);
//ONNXRUNTIME_API_STATUS(ONNXRuntimeInferenceSessionGetOutputType, _In_ ONNXSessionPtr sess, _Out_ PONNXType* out);

/**
 * Get the n-th value from the List
 * \param index starts from zero
 */
ONNXRUNTIME_API(ONNXValuePtr, ONNXRuntimeONNXValueListGetNthValue, _In_ ONNXValueListPtr list, size_t index);

ONNXRUNTIME_API_STATUS(ONNXRuntimeTensorProtoToONNXValue, _Inout_ ONNXRuntimeAllocator* allocator, _In_ const void* input, int input_len, _Out_ ONNXValuePtr* out);

#ifdef __cplusplus
}
#endif

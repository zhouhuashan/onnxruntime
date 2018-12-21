using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{

    internal enum LogLevel
    {
        Verbose = 0,
        Info = 1,
        Warning = 2,
        Error = 3,
        Fatal = 4
    }

    /// <summary>
    /// Enum conresponding to native onnxruntime error codes. Must be in sync with the native API
    /// </summary>
    internal enum ErrorCode
    {
        Ok = 0,
        Fail = 1,
        InvalidArgument = 2,
        NoSuchFile = 3,
        NoModel = 4,
        EngineError = 5,
        RuntimeException = 6,
        InvalidProtobuf = 7,
        ModelLoaded = 8,
        NotImplemented = 9,
        InvalidGraph = 10,
        ShapeInferenceNotRegistered = 11,
        RequirementNotRegistered = 12
    }

    internal enum TensorElementType
    {
        Float = 1,
        UInt8 = 2,
        Int8 = 3,
        UInt16 = 4,
        Int16 = 5,
        Int32 = 6,
        Int64 = 7,
        String = 8,
        Bool = 9,
        Float16 = 10,
        Double = 11,
        UInt32 = 12,
        UInt64 = 13,
        Complex64 = 14,
        Complex128 = 15,
        BFloat16 = 16,
        DataTypeMax = 17
    }

}

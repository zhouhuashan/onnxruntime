// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{

    /// <summary>
    /// The Exception that is thrown for errors related ton OnnxRuntime
    /// </summary>
    public class OnnxRuntimeException: Exception
    {
        private static Dictionary<ErrorCode, string> errorCodeToString = new Dictionary<ErrorCode, string>()
        {
            { ErrorCode.Ok, "Ok" },
            { ErrorCode.Fail, "Fail" },
            { ErrorCode.InvalidArgument, "InvalidArgument"} ,
            { ErrorCode.NoSuchFile, "NoSuchFile" },
            { ErrorCode.NoModel, "NoModel" },
            { ErrorCode.EngineError, "EngineError" },
            { ErrorCode.RuntimeException, "RuntimeException" },
            { ErrorCode.InvalidProtobuf, "InvalidProtobuf" },
            { ErrorCode.ModelLoaded, "ModelLoaded" },
            { ErrorCode.NotImplemented, "NotImplemented" },
            { ErrorCode.InvalidGraph, "InvalidGraph" },
            { ErrorCode.ShapeInferenceNotRegistered, "ShapeInferenceNotRegistered" },
            { ErrorCode.RequirementNotRegistered, "RequirementNotRegistered" }
        };

        internal OnnxRuntimeException(ErrorCode errorCode, string message)
            :base("[ErrorCode:" + errorCodeToString[errorCode] + "] " + message)
        {
        }
    }


}

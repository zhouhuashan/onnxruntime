"""
*ONNX Runtime* is a critical component for platforms that 
enables high-performance evaluation of trained machine learning (ML)
models while keeping resource usage low. 
Building on Microsoft's dedication to the 
`Open Neural Network Exchange (ONNX) <https://onnx.ai/>`_
community, it supports traditional ML models as well 
as Deep Learning algorithms in the
`ONNX-ML format <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_.
"""
__version__ = "0.1.0"
__author__ = "Microsoft"
__url__ = "https://github.com/onnx/onnx-runtime"

from onnx_runtime.python.session import InferenceSession
from onnx_runtime.python._pybind_state import RunOptions
from onnx_runtime.python._pybind_state import SessionOptions
from onnx_runtime.python import datasets

# @package _pybind_state
# Module onnx_runtime.python._pybind_state
import sys
import os
import warnings

try:
    from onnx_runtime.python.onnx_runtime_pybind11_state import *  # noqa
except ImportError as e:
    warnings.warn("Cannot load onnx_runtime.python. Error: '{0}'".format(str(e)))

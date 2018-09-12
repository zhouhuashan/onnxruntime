# @package _pybind_state
# Module onnxruntime.python._pybind_state
import sys
import os
import warnings

try:
    from onnxruntime.python.onnxruntime_pybind11_state import *  # noqa
except ImportError as e:
    warnings.warn("Cannot load onnxruntime.python. Error: '{0}'".format(str(e)))

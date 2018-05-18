## @package session
# Module lotus.python.session
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import logging
import ctypes
import os

logger = logging.getLogger(__name__)
from lotus.python import _pybind_state as C

class InferenceSession:
  def __init__(self, path):
    self._runtime = C.InferenceSession(
      C.get_session_initializer(), C.get_session_initializer())
    self._runtime.load_model(path)
    self._inputs_meta = self._runtime.inputs_meta
    self._outputs_meta = self._runtime.outputs_meta

  def get_inputs(self) :
    return self._inputs_meta

  def get_outputs(self) :
    return self._outputs_meta

  def run(self, output_names, input_feed) :
    if not output_names:
      output_names = [ output.name for output in self._outputs_meta ]
    return self._runtime.run(output_names, input_feed)

if __name__ == "__main__":
  if not 'InferenceSession' in dir(C):
    raise RuntimeError('Failed to bind the native module -lotus_pybind11_state.pyd.')


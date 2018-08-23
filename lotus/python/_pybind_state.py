# @package _pybind_state
# Module lotus.python._pybind_state
import sys
import os
import logging

logger = logging.getLogger(__name__)

try:
    from lotus.python.lotus_pybind11_state import *  # noqa
except ImportError as e:
    logging.critical(
        'Cannot load lotus.python. Error: {0}'.format(str(e)))

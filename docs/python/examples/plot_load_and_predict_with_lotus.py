"""
.. _l-example-simple-usage:

Load and predict with ONNX Runtime and a very simple model
==========================================================

This example demonstrates how to load a model and compute
the output for an input vector. It also shows how to
retrieve the definition of its inputs and outputs.

Basic usage
+++++++++++
"""

import onnx_runtime
import numpy
from onnx_runtime.python.datasets import get_example

#########################
# Let's load a very simple model.

example1 = get_example("mul_1.pb")
sess = onnx_runtime.InferenceSession(example1)

#########################
# Let's see the input name and shape.

input_name = sess.get_inputs()[0].name
print(input_name)
input_shape = sess.get_inputs()[0].shape
print(input_shape)

#########################
# Let's see the output name and shape.

output_name = sess.get_outputs()[0].name
print(output_name)  
output_shape = sess.get_outputs()[0].shape
print(output_shape)

#########################
# Let's compute its outputs (or predictions if it is a machine learned model).

x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
res = sess.run([output_name], {input_name: x})
print(res)

#########################
# Basic errors
# ++++++++++++
#
# What happens if the dimension or the name are not right.
# First, bad type.

try:
    x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float64)
    sess.run([output_name], {input_name: x})
except Exception as e:
    print("{0}: {1}".format(type(e), e))
    
#########################
# Then bad dimension.

try:
    x = numpy.array([[1.0, 2.0, 1.0], [3.0, 4.0, 5.0]], dtype=numpy.float32)
    sess.run([output_name], {input_name: x})
except Exception as e:
    print("{0}: {1}".format(type(e), e))

#########################
# Then bad names.

try:
    x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float64)
    sess.run(["whatever"], {input_name: x})
except Exception as e:
    print("{0}: {1}".format(type(e), e))

try:
    x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float64)
    sess.run([output_name], {"whatever": x})
except Exception as e:
    print("{0}: {1}".format(type(e), e))

try:
    x = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float64)
    sess.run(output_name, {input_name: x})
except Exception as e:
    print("{0}: {1}".format(type(e), e))


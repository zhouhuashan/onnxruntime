
===================
Improve ONNX models
===================

*onnxruntime* only computes the outputs of one
ONNX pipeline converted. This page explores
ways to modify a pipeline to produce different outputs or
the same outputs but faster.

.. contents::
    :local:

Performance
===========

Module `onnx <https://github.com/onnx/onnx>`_
provides helper to optimize the model performance
while keeping the same outputs. More information
can be found at
`Optimizing an ONNX Model <https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#optimizing-an-onnx-model>`_.

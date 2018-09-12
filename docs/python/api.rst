
===
API
===

Summary of public function and classes exposed
in *onnxruntime*.

.. contents::
    :local:

Device
======

The package is compiled for a specific device, GPU or CPU.
The CPU comes with several available optimisations
such as MKL (Math Kernel Libary). The following function
indicates the chosen option:

.. autofunction:: onnxruntime.get_device

Examples and datasets
=====================

The package contains a few models stored in ONNX format
used in the documentation. They don't need to be downloaded
as they are installed with the package.

.. autofunction:: onnxruntime.datasets.get_example

Load and run a model
====================

*onnxruntime* reads a model saved in ONNX format but
uses its own internal structure to hold the model in memory.
The main class *InferenceSession* wraps these functionalities
in a single place.

.. autoclass:: onnxruntime.InferenceSession
    :members:

.. autoclass:: onnxruntime.RunOptions
    :members:

.. autoclass:: onnxruntime.SessionOptions
    :members:

Backend
=======

The runtime is also available through the 
`ONNX backend API <https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md>`_
with the following functions.

.. autofunction:: onnxruntime.backend.is_compatible

.. autofunction:: onnxruntime.backend.prepare

.. autofunction:: onnxruntime.backend.run

.. autofunction:: onnxruntime.backend.supports_device

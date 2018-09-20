
Python Bindings for ONNX Runtime (Preview)
==========================================

ONNX Runtime (Preview) enables high-performance evaluation of trained machine learning (ML)
models while keeping resource usage low. 
Building on Microsoft's dedication to the 
`Open Neural Network Exchange (ONNX) <https://onnx.ai/>`_
community, it supports traditional ML models as well 
as Deep Learning algorithms in the
`ONNX-ML format <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_.

.. only:: html

    .. toctree::
        :maxdepth: 1

        tutorial
        api
        auto_examples/index
        
    :ref:`genindex`

.. only:: md

    .. toctree::
        :maxdepth: 1
        :caption: Contents:

        tutorial
        api
        examples_md

The core library is implemented in C++.
*ONNX Runtime* is available on 
PyPi for Linux Ubuntu 16.04, Python 3.5+ for both
`CPU <https://pypi.org/project/onnxruntime/>`_ and
`GPU <https://pypi.org/project/onnxruntime-gpu/>`_.
This example demonstrates a simple prediction for an
`ONNX-ML format <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_
model:

::

    import onnxruntime as rt
    sess = rt.InferenceSession("model.onnx")
    input_name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {input_name: X})




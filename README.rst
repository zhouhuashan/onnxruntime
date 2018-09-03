ONNX Runtime
============

ONNX Runtime is a critical component for platforms that 
enables high-performance evaluation of trained machine learning (ML)
models while keeping resource usage low. 
Building on Microsoft's dedication to the 
`Open Neural Network Exchange (ONNX) <https://onnx.ai/>`_
community, it supports traditional ML models as well 
as Deep Learning algorithms in the ONNX-ML format.

Example
-------

The following example demonstrates an end-to-end example
in a very common scenario. A model is trained with *scikit-learn*
but it has to run very fast in a optimized environment.
The model is then converted into ONNX format and ONNX Runtime
replaces *scikit-learn* to compute the predictions.

::

    # Train a model.
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForest
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForest()
    clr.fit(X_train, y_train)

    # Convert into ONNX format with onnxmltools
    from onnxmltools import convert_sklearn
    from onnxmltools.utils import save_model
    from onnxmltools.convert.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([1, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    save_model(onx, "rf_iris.onnx")

    # Compute the prediction with ONNX Runtime
    import onnx_runtime
    import numpy
    sess = onnx_runtime.InferenceSession("rf_iris.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]   

Supported environments
----------------------

+--------------------------------+--------------+-------------+--------------------------------------------+
| OS                             | Supports CPU | Supports GPU| Notes                                      | 
|================================+==============+=============+============================================+
| Windows 10                     | YES          | YES         | Must use VS 2017 or the latest VS2015      |
| Windows 10 Subsystem for Linux | YES          | NO          |                                            |
| Ubuntu 16.x                    | YES          | YES         |                                            |
| Ubuntu 17.x                    | YES          | YES         |                                            |
| Ubuntu 18.x                    | YES          | UNKNOWN     | No CUDA package from Nvidia                |
| Fedora 24                      | YES          | YES         |                                            |
| Fedora 25                      | YES          | YES         |                                            |
| Fedora 26                      | YES          | YES         |                                            |
| Fedora 27                      | YES          | YES         |                                            |
| Fedora 28                      | YES          | NO          | Cannot build GPU kernels but can run them  |
+--------------------------------+--------------+-------------+--------------------------------------------+

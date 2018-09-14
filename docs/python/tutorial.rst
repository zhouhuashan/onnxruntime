
========
Tutorial
========

*onnxruntime* provide a natural backend for *onnx*
available optimized for CPU or GPU devices.
It provides an easy way to run machine learned models
with no dependencies. Machine learning framework are
optimized for batch training and not necessarily
one off prediction which is the typical scenario
in many websites. *onnxruntime* addresses this
situation. The scenario is usually the following:

1. A model is trained with you favorite framework and
   is composed of a single pipeline.
2. A tool converts this pipeline into ONNX format,
   *onnmltools*, *winmltools*, *coremltools* or any
   specific extension for deep learning,
   such as *onnx-torch*, *onnx-tensorflow*.
3. The model is loaded with *onnxruntime*
   and available to be run for new predictions.


In this tutorial, we will briefly create a 
pipeline with *scikit-learn*, convert it into
ONNX format and run the first predictions.

Step 1: create a machine learning pipeline
++++++++++++++++++++++++++++++++++++++++++

We use the famous iris datasets.

::

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    from sklearn.linear_model import LogisticRegression
    clr = LogisticRegression()
    clr.fit(X_train, y_train)

Step 2: convert the model into ONNX format
++++++++++++++++++++++++++++++++++++++++++

*ONNX* is the format used to describe the machine learned model.
It provides a list of many mathematical functions used
in most of the models. A tree is a function, a random forest
is a sum of tree functions... Fortunately, there exists a tool
to convert many kinds of models into ONNX:
`onnxmltools <https://github.com/onnx/onnxmltools>`_.

::

    from onnxmltools import convert_sklearn
    from onnxmltools.utils import save_model
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([1, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    save_model(onx, "logreg_iris.onnx")

Step 3: compute the predictions
+++++++++++++++++++++++++++++++

The machine learning library used to train the model
cannot be used to compute the predictions with this new
format. It requires a dedicated runtime, it is also 
is usually optimized for a specific device
and can be available in other languages than *python*.
*onnxruntime* is optimized for CPU, GPU, available in
*python* and *C*.

::

    import onnxruntime as onnxrt
    sess = onnxrt.InferenceSession("logreg_iris.onnx")
    input_name = sess.get_inputs()[0].name
    
    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

*onnxruntime* requires the inputs to be converted
into *float* which allows faster computations in CPU and GPU.
That might create some discrepencies between the predictions
with the original model but they should be small.


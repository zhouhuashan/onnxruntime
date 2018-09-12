
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


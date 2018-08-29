"""
Train, convert and predict with Lotus
=====================================

This example demonstrates an end to end scenario
starting with the training of a scikit-learn pipeline
which takes as inputs not a regular vector but a
dictionary ``{ int: float }`` as its first step is a
`DictVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html>`_.

.. contents::
    :local:

Train a pipline
+++++++++++++++

The first step consists in retrieving the boston datset.
"""

# input()
import lotus
sess = lotus.InferenceSession("vectorize.onnx")
#stop

import pandas
from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data, boston.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_dict = pandas.DataFrame(X_train[:,1:]).T.to_dict().values()
X_test_dict = pandas.DataFrame(X_test[:,1:]).T.to_dict().values()

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
pipe = DictVectorizer(sparse=False)
pipe.fit(X_train_dict, y_train)
print(pipe.transform(X_train_dict)[:5])

from onnxmltools import convert_sklearn
from onnxmltools.utils import save_model
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType, DictionaryType

initial_type = [('float_input', DictionaryType(Int64TensorType([1]), FloatTensorType([1])))]
onx = convert_sklearn(pipe, initial_types=initial_type)
save_model(onx, "vectorize.onnx")

import lotus
#fails here
sess = lotus.InferenceSession("vectorize.onnx")

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

import numpy
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)



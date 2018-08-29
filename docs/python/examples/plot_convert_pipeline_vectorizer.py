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
import pandas
from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data, boston.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_dict = pandas.DataFrame(X_train[:,1:]).T.to_dict().values()
X_test_dict = pandas.DataFrame(X_test[:,1:]).T.to_dict().values()

if False: # TODO: remove
    
    ####################################
    # We create a pipeline.

    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.feature_extraction import DictVectorizer
    pipe = make_pipeline(
                DictVectorizer(sparse=False),
                GradientBoostingRegressor())
                
    pipe.fit(X_train_dict, y_train)

    ####################################
    # We compute the prediction on the test set
    # and we show the confusion matrix.
    from sklearn.metrics import r2_score

    pred = pipe.predict(X_test_dict)
    print(r2_score(y_test, pred))

    ####################################
    # Conversion to ONNX format
    # +++++++++++++++++++++++++
    #
    # We use module 
    # `onnxmltools <https://github.com/onnx/onnxmltools>`_
    # to convert the model into ONNX format.

    from onnxmltools import convert_sklearn
    from onnxmltools.utils import save_model
    from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType, DictionaryType

    # initial_type = [('float_input', DictionaryType(Int64TensorType([1]), FloatTensorType([])))]
    initial_type = [('float_input', DictionaryType(Int64TensorType([1]), FloatTensorType([])))]
    onx = convert_sklearn(pipe, initial_types=initial_type)
    save_model(onx, "pipeline_vectorize.onnx")

##################################
# We load the model with Lotus and look at
# its input and output.
import lotus
sess = lotus.InferenceSession("pipeline_vectorize.onnx")

import numpy
print(dir(numpy.array([[0,1]])))

print("input name='{}' and shape={}".format(sess.get_inputs()[0].name, sess.get_inputs()[0].shape))
print("output name='{}' and shape={}".format(sess.get_outputs()[0].name, sess.get_outputs()[0].shape))

##################################
# We compute the predictions and compare them
# to the model's ones.

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

import numpy
# input()
pred_onx = sess.run([label_name], {input_name: X_test_dict})[0]
print(confusion_matrix(pred, pred_onx))

#####################################
# Draw a pipeline with ONNX
# +++++++++++++++++++++++++
#
# We use `net_drawer.py <https://github.com/onnx/onnx/blob/master/onnx/tools/net_drawer.py>`_
# included in *onnx* package.
# We first use *onnx* to load the model.

from onnx import ModelProto
model = ModelProto()
with open("pipeline_vectorize.onnx", 'rb') as fid:
    content = fid.read()
    model.ParseFromString(content)

###################################
# We convert it into a graph.
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
pydot_graph = GetPydotGraph(model.graph, name=model.graph.name, rankdir="LR",
                            node_producer=GetOpNodeProducer("docstring"))
pydot_graph.write_dot("graph.dot")

#######################################
# Then into an image
os.system('dot -O -Tpng graph.dot')

################################
# Which we display...
import matplotlib.pyplot as plt
image = plt.imread("graph.png")
plt.imshow(image)




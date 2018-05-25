from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest
import lotus

class TestInferenceSession(unittest.TestCase):

    def testRunModel(self):
        sess = lotus.InferenceSession("testdata/mul_1.pb")
        input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 2])
        res = sess.run([output_name], {input_name: input})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModel2(self):
        sess = lotus.InferenceSession("testdata/matmul_1.pb")
        input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 1])
        res = sess.run([output_name], {input_name: input})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModelSymbolicInput(self):
        sess = lotus.InferenceSession("testdata/matmul_2.pb")
        input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        # Input X has an unknown dimension.
        self.assertEqual(input_shape, [None, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        # Output X has an unknown dimension.
        self.assertEqual(output_shape, [None, 1])
        res = sess.run([output_name], {input_name: input})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

if __name__ == '__main__':
    unittest.main()

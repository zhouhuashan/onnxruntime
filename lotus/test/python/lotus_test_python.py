from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest
import lotus

class TestInferenceSession(unittest.TestCase):

    def testRunModel(self):
        m = lotus.InferenceSession("testdata/mul_1.pb")
        input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = m.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        shape = m.get_inputs()[0].shape
        self.assertEqual(shape, [3, 2])
        output_name = m.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        res = m.run([output_name], {input_name: input})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

if __name__ == '__main__':
    unittest.main()
